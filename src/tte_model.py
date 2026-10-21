"""
Estimación de tiempo de viaje origen-destino (OD TTE) entre ciudades a partir
de trayectorias GPS.

Ideas que combina el modelo:
  TransferTraj (NeurIPS 2025)  celdas descritas por modalidades (patrón horario,
                               POI, red vial) y features espaciales relativas,
                               sin coordenadas absolutas.
  OOD4TTE (KDD 2026)           separación ruta (T) / contexto (C), entornos
                               latentes y compuertas invariantes.
  TSJT (AAAI 2025)             entrenamiento en una etapa sobre objetivo +
                               fuentes con Target-Skewed Backward y banco de
                               prompts compartido (NPM).
  GNNUI (2025)                 estado espaciotemporal por celda con
                               interpolación enmascarada (ver st_attention.py).
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import st_attention as S
import trajectory_loader as tl
import urban_context as uc


# RUTA (T): geometría del par O-D y del corredor recto. Nada del recorrido real.
ROUTE_NAMES = [
    'dist_norm', 'dist_log', 'bearing_sin', 'bearing_cos',
    'o_radial', 'd_radial', 'radial_delta',
    'inter_per_km',
    'corr_arterial', 'corr_colectora', 'corr_local', 'corr_peatonal',
]
# CONTEXTO (C): cuándo y en qué condiciones ocurre el viaje.
CTX_NAMES = [
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend',
    'call_A', 'call_B', 'call_C', 'day_A', 'day_B', 'day_C',
    'signal_per_km', 'friction', 'poi_corridor_q', 'ctx_mask',
]
F_ROUTE = len(ROUTE_NAMES)
F_CTX = len(CTX_NAMES)
SL_ROUTE = slice(0, F_ROUTE)
SL_CTX = slice(F_ROUTE, F_ROUTE + F_CTX)
I_DIST_NORM = ROUTE_NAMES.index('dist_norm')
I_DIST_LOG = ROUTE_NAMES.index('dist_log')

CELL_POI_NAMES = list(uc.NODE_POI_NAMES)
CELL_ROAD_NAMES = list(uc.NODE_ROAD_NAMES)
N_HOURS = 24


@dataclass
class STTensors:
    """Semana típica de una ciudad para el módulo espaciotemporal.

    spd/obs apilan los tensores construidos sin cada pliegue y, al final, el
    construido con todo el train: el índice -1 es siempre el completo.
    """
    feat: torch.Tensor     # [168, C, F_ST_RAW] canales de entrada
    spd: torch.Tensor      # [n_folds + 1, 168, C] velocidad media estandarizada
    obs: torch.Tensor      # [n_folds + 1, 168, C] 1 si la celda-hora tiene viajes
    how: torch.Tensor      # [N] hora de la semana de cada viaje
    fold: torch.Tensor     # [N] pliegue de cada viaje de train, -1 para el resto
    P_fwd: torch.Tensor    # transiciones del grafo dirigido (dispersas)
    P_bwd: torch.Tensor

    @property
    def coverage(self):
        return float(self.obs[-1].mean())

    def to(self, device):
        return STTensors(*(getattr(self, f).to(device) for f in self.__dataclass_fields__))


@dataclass
class CityData:
    city: str
    feats: torch.Tensor            # [N, F_ROUTE + F_CTX]
    o_cell: torch.Tensor           # [N]
    d_cell: torch.Tensor           # [N]
    times: torch.Tensor            # [N] log-tiempo normalizado
    raw_times: torch.Tensor        # [N] segundos
    cell_poi: torch.Tensor         # [C, F_POI]
    cell_road: torch.Tensor        # [C, F_ROAD]
    cell_mask: torch.Tensor        # [C, 2] disponibilidad POI / vial
    cell_pattern: torch.Tensor     # [C, 24 + 2]
    edge_index: tuple
    edge_weight: torch.Tensor
    st: STTensors
    train_idx: List[int] = field(default_factory=list)
    val_idx: List[int] = field(default_factory=list)
    test_idx: List[int] = field(default_factory=list)
    time_log_mean: float = 0.0
    time_log_std: float = 1.0
    # corredor O-D (solo con --corridor, ver st_attention.attach_corridors)
    corr_cells: Optional[torch.Tensor] = None
    corr_mask: Optional[torch.Tensor] = None
    corr_pos: Optional[torch.Tensor] = None
    pair_of_trip: Optional[torch.Tensor] = None

    @property
    def n_cells(self):
        return self.cell_poi.size(0)

    def to(self, device):
        """Mueve lo que se usa por ciudad. Los tensores por viaje (feats,
        o_cell, ...) se quedan en CPU y viajan por lotes."""
        for name in ('cell_poi', 'cell_road', 'cell_mask', 'cell_pattern', 'edge_weight',
                     'corr_cells', 'corr_mask', 'corr_pos', 'pair_of_trip'):
            if getattr(self, name) is not None:
                setattr(self, name, getattr(self, name).to(device))
        self.edge_index = tuple(e.to(device) for e in self.edge_index)
        self.st = self.st.to(device)
        return self


# ---------------------------------------------------------------------------
# Features y tensores
# ---------------------------------------------------------------------------

def hour_and_dow(ts):
    """Hora del día y día de la semana con lunes = 0 (el 1-1-1970 fue jueves)."""
    return (ts // 3600) % 24, ((ts // 86400) + 3) % 7


def _cyclic(x, period):
    a = 2 * np.pi * x / period
    return np.sin(a), np.cos(a)


def _trip_vector(t, city, dist_mean, dist_std):
    """Vector [RUTA | CONTEXTO] en el orden de ROUTE_NAMES + CTX_NAMES.

    Lo que no se calcula aquí (contexto del corredor) se toma del propio viaje.
    """
    d = t['od_dist']
    o_rad = min(uc.haversine(t['o_lat'], t['o_lon'], city.lat_c, city.lon_c) / city.radius, 2.0)
    d_rad = min(uc.haversine(t['d_lat'], t['d_lon'], city.lat_c, city.lon_c) / city.radius, 2.0)
    hour, dow = hour_and_dow(t['timestamp'])
    f = {
        'dist_norm': (d - dist_mean) / dist_std,
        'dist_log': math.log1p(d) / 10.0,
        'o_radial': o_rad, 'd_radial': d_rad, 'radial_delta': d_rad - o_rad,
        'is_weekend': float(dow >= 5),
        'ctx_mask': float(bool(t.get('has_ctx'))),
    }
    f['bearing_sin'], f['bearing_cos'] = _cyclic(
        uc.bearing(t['o_lat'], t['o_lon'], t['d_lat'], t['d_lon']), 360)
    f['hour_sin'], f['hour_cos'] = _cyclic(hour, 24)
    f['dow_sin'], f['dow_cos'] = _cyclic(dow, 7)
    f.update({f'call_{c}': float(t['call_type'] == c) for c in tl.CALL_TYPES})
    f.update({f'day_{c}': float(t['day_type'] == c) for c in tl.DAY_TYPES})
    return [f[n] if n in f else float(t.get(n, 0.0)) for n in ROUTE_NAMES + CTX_NAMES]


def _cell_patterns(city, train_idx):
    """Perfil horario de demanda + intensidad O/D de cada celda (Query del NPM).

    Solo con viajes de entrenamiento, para no meter el test en los prompts.
    """
    C = city.grid.n_cells
    prof = np.zeros((C, N_HOURS))
    n_o, n_d = np.zeros(C), np.zeros(C)
    for i in train_idx:
        t = city.trips[i]
        h = hour_and_dow(t['timestamp'])[0]
        prof[t['o_cell'], h] += 1
        prof[t['d_cell'], h] += 1
        n_o[t['o_cell']] += 1
        n_d[t['d_cell']] += 1
    prof /= np.maximum(prof.sum(axis=1, keepdims=True), 1.0)
    return np.concatenate([prof, uc.quantile_rank(n_o)[:, None],
                           uc.quantile_rank(n_d)[:, None]], axis=1).astype(np.float32)


def build_st_tensors(city, train_idx, graph_k=8, n_folds=2, seed=0):
    """Semana típica [168, C, canales] con la velocidad O-D como variable objetivo.

    Se usa una semana típica y no el calendario real porque el split es
    cronológico: el periodo de prueba no tendría datos.

    La velocidad de una celda-hora sale de los tiempos de entrenamiento, así
    que un viaje de train podría leer su propia etiqueta. Se evita con
    codificación fuera de pliegue: cada viaje de train lee el tensor construido
    sin su pliegue; validación y prueba leen el tensor completo.
    """
    C = city.grid.n_cells
    trips = city.trips
    tr = np.asarray(train_idx, dtype=np.int64)

    hd = np.array([hour_and_dow(t['timestamp']) for t in trips], dtype=np.int64)
    how = hd[:, 1] * 24 + hd[:, 0]                  # hora de la semana en [0, 168)
    o = np.array([t['o_cell'] for t in trips], dtype=np.int64)
    d = np.array([t['d_cell'] for t in trips], dtype=np.int64)
    lsp = np.log(np.maximum([t['od_dist'] / t['travel_time'] for t in trips], 1e-3))
    z = (lsp - lsp[tr].mean()) / (lsp[tr].std() + 1e-8)

    fold = np.full(len(trips), -1, dtype=np.int64)
    fold[tr] = np.random.RandomState(seed).randint(0, n_folds, size=len(tr))

    n_o = np.zeros((S.N_HOW, C))
    n_d = np.zeros((S.N_HOW, C))
    np.add.at(n_o, (how[tr], o[tr]), 1.0)
    np.add.at(n_d, (how[tr], d[tr]), 1.0)
    lo, ld = np.log1p(n_o), np.log1p(n_d)
    lo /= max(lo.max(), 1e-8)
    ld /= max(ld.max(), 1e-8)

    hrs = np.arange(S.N_HOW)
    tchan = np.stack([*_cyclic(hrs % 24, 24), *_cyclic(hrs // 24, 7)], 1)
    feat = np.concatenate([lo[..., None], ld[..., None],
                           np.broadcast_to(tchan[:, None, :], (S.N_HOW, C, 4))], -1)

    def speed(sel):
        s_ = np.zeros((S.N_HOW, C))
        c_ = np.zeros((S.N_HOW, C))
        for cells in (o, d):
            np.add.at(s_, (how[sel], cells[sel]), z[sel])
            np.add.at(c_, (how[sel], cells[sel]), 1.0)
        return (s_ / np.maximum(c_, 1.0)).astype(np.float32), (c_ > 0).astype(np.float32)

    # sin cada pliegue y, al final, con todo el train
    spd, obs = zip(*[speed(tr[fold[tr] != f]) for f in range(n_folds)], speed(tr))
    src, dst = city.build_directed_adjacency(tr, k=graph_k)
    P_fwd, P_bwd = S.transition_matrices(src, dst, C)
    return STTensors(feat=torch.tensor(feat, dtype=torch.float32),
                     spd=torch.tensor(np.stack(spd)), obs=torch.tensor(np.stack(obs)),
                     how=torch.tensor(how), fold=torch.tensor(fold),
                     P_fwd=P_fwd, P_bwd=P_bwd)


def build_city_data(city, norms, split='chronological', split_kwargs=None,
                    graph_k=8, corridor_k=None, target_fraction=None):
    """Tensores de una ciudad. target_fraction recorta el train ANTES de
    construir grafo, patrones y tensores ST, para que el régimen few-shot no
    vea información de los viajes descartados."""
    dist_mean, dist_std, time_log_mean, time_log_std = norms
    split_kwargs = split_kwargs or {}
    if split == 'chronological':
        tr, va, te = city.chronological_split(**split_kwargs)
    elif split == 'ood_distance':
        tr, te = city.ood_split_by_distance(**split_kwargs)
        cut = int(len(tr) * 0.9)
        tr, va = tr[:cut], tr[cut:]
    else:
        raise ValueError(f"split desconocido: {split!r}")
    if target_fraction:
        tr = tr[:max(int(len(tr) * target_fraction), 8)]

    feats = np.array([_trip_vector(t, city, dist_mean, dist_std) for t in city.trips],
                     dtype=np.float32)
    raw = np.array([t['travel_time'] for t in city.trips], dtype=np.float32)
    times = (np.log(np.maximum(raw, 1.0)) - time_log_mean) / time_log_std

    cells = city.cells
    poi = np.array([[c.get(n, 0.0) for n in CELL_POI_NAMES] for c in cells], dtype=np.float32)
    road = np.array([[c.get(n, 0.0) for n in CELL_ROAD_NAMES] for c in cells], dtype=np.float32)
    mask = np.array([[float(bool(c.get('has_poi_ctx'))), float(bool(c.get('has_road_ctx')))]
                     for c in cells], dtype=np.float32)

    (src, dst), val = city.build_region_graph(tr, k=graph_k)

    cd = CityData(
        city=city.city,
        feats=torch.tensor(feats),
        o_cell=torch.tensor([t['o_cell'] for t in city.trips], dtype=torch.long),
        d_cell=torch.tensor([t['d_cell'] for t in city.trips], dtype=torch.long),
        times=torch.tensor(times, dtype=torch.float32),
        raw_times=torch.tensor(raw),
        cell_poi=torch.tensor(poi), cell_road=torch.tensor(road),
        cell_mask=torch.tensor(mask),
        cell_pattern=torch.tensor(_cell_patterns(city, tr)),
        edge_index=(torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)),
        edge_weight=torch.tensor(val, dtype=torch.float32),
        st=build_st_tensors(city, tr, graph_k),
        train_idx=tr, val_idx=va, test_idx=te,
        time_log_mean=time_log_mean, time_log_std=time_log_std)
    if corridor_k:
        S.attach_corridors(cd, city, K=corridor_k)
    return cd


def global_norms(cities):
    """Normalización de distancia y log-tiempo compartida por todas las ciudades."""
    d = np.array([t['od_dist'] for c in cities for t in c.trips])
    lt = np.log(np.maximum([t['travel_time'] for c in cities for t in c.trips], 1.0))
    return float(d.mean()), float(d.std() + 1e-8), float(lt.mean()), float(lt.std() + 1e-8)


def train_dataset(cd):
    """Viajes de train como (feats, o_cell, d_cell, times, índice del viaje)."""
    idx = torch.tensor(cd.train_idx, dtype=torch.long)
    return TensorDataset(cd.feats[idx], cd.o_cell[idx], cd.d_cell[idx], cd.times[idx], idx)


# ---------------------------------------------------------------------------
# Módulos
# ---------------------------------------------------------------------------

class CellEncoder(nn.Module):
    """Fusión por atención de las modalidades de la celda.

    Una modalidad sin datos (ciudad sin OSM) usa un embedding aprendido en vez
    de ceros, que el modelo leería como "zona sin nada alrededor".
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        def enc(n):
            return nn.Sequential(nn.Linear(n, dim), nn.ReLU(), nn.LayerNorm(dim))
        self.enc_poi = enc(len(CELL_POI_NAMES))
        self.enc_road = enc(len(CELL_ROAD_NAMES))
        self.enc_pat = enc(N_HOURS + 2)
        self.missing = nn.Parameter(torch.randn(2, dim) * 0.02)
        self.attn = nn.Linear(dim, 1)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, dim), nn.LayerNorm(dim))

    def forward(self, poi, road, mask, pattern):
        e_pat = self.enc_pat(pattern)
        e_poi = torch.where(mask[:, :1] > 0.5, self.enc_poi(poi),
                            self.missing[0].expand_as(e_pat))
        e_road = torch.where(mask[:, 1:2] > 0.5, self.enc_road(road),
                             self.missing[1].expand_as(e_pat))
        tok = torch.stack([e_pat, e_poi, e_road], dim=1)
        a = torch.softmax(self.attn(tok), dim=1)
        return self.out((tok * a).sum(1))


class RegionMessagePassing(nn.Module):
    """h' = LN(h + ReLU(W_s h + W_n Â h)); el residuo evita que las celdas se igualen."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.w_self = nn.Linear(dim, dim)
        self.w_nbr = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_weight):
        src, dst = edge_index
        agg = torch.zeros_like(h).index_add_(0, dst, h[src] * edge_weight.unsqueeze(1))
        return self.norm(h + self.drop(F.relu(self.w_self(h) + self.w_nbr(agg))))


class NodePromptingModule(nn.Module):
    """Banco de prompts compartido por todas las ciudades (TSJT).

    Celdas con patrones de demanda parecidos recuperan prompts parecidos,
    aunque sean de ciudades distintas.
    """

    def __init__(self, dim, bank_size=64):
        super().__init__()
        self.bank = nn.Parameter(torch.randn(bank_size, dim) * 0.02)
        self.q = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, h_cell):
        att = torch.softmax(self.q(h_cell) @ self.bank.T * self.scale, dim=-1)
        return h_cell + att @ self.bank


class EnvironmentInference(nn.Module):
    """Pertenencia blanda a V entornos de tráfico latentes (OOD4TTE)."""

    def __init__(self, in_dim, n_env=8, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_env))

    def forward(self, phi, psi):
        return torch.softmax(self.net(torch.cat([phi, psi], dim=1)), dim=1)


class InvariantGate(nn.Module):
    """Conserva el k% de dimensiones más estables entre entornos (OOD4TTE).

    La varianza se guarda como EMA para que la máscara en inferencia no dependa
    del lote.
    """

    def __init__(self, dim, k_ratio=0.5, momentum=0.05):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.k_ratio = k_ratio
        self.momentum = momentum
        self.register_buffer('var_ema', torch.ones(dim))

    def forward(self, z):
        p = self.proj(z)
        if self.training and z.size(0) > 2:
            perm = torch.randperm(z.size(0), device=z.device)
            self.var_ema.mul_(1 - self.momentum).add_(
                self.momentum * (p - p[perm]).var(0, unbiased=False).detach())
        k = max(int(self.var_ema.numel() * self.k_ratio), 1)
        thr = torch.topk(self.var_ema, k, largest=False).values.max()
        return z * torch.sigmoid(p) * (self.var_ema <= thr).float()


def _double_center(x):
    d = torch.cdist(x, x)
    return d - d.mean(0, keepdim=True) - d.mean(1, keepdim=True) + d.mean()


def partial_distance_corr(a_c, a_t, a_e, eps=1e-8):
    """Correlación de distancia parcial entre C y T dado E (Székely & Rizzo)."""
    inner = lambda a, b: (a * b).mean()
    de = inner(a_e, a_e) + eps
    p_c = a_c - inner(a_c, a_e) / de * a_e
    p_t = a_t - inner(a_t, a_e) / de * a_e
    den = inner(p_c, p_c).clamp(min=eps).sqrt() * inner(p_t, p_t).clamp(min=eps).sqrt()
    return inner(p_c, p_t).abs() / (den + eps)


class TravelTimeModel(nn.Module):
    """
        celdas --> CellEncoder --> grafo de regiones --> NPM --> h
        [h_o | h_d | RUTA]            --> Ψ   (ruta)
        [CONTEXTO | estado ST de O,D] --> Φ   (contexto)
        cabeza(gate(Φ) + gate(Ψ))     --> log-tiempo normalizado

    st_kind   'dgcn' (UrbanSTGNN por ventanas) o 'tsa' (Crossformer/iTransformer).
    pace      y = log d + ritmo medio + residuo; el residuo no ve la distancia,
              que en ood_distance cae fuera del rango de entrenamiento.
    corridor  añade el log-ritmo del corredor O-D (implica pace).
    """

    def __init__(self, dim=64, hidden=128, dropout=0.1, bank_size=64, n_env=8,
                 invariant_ratio=0.5, n_graph_layers=2, st_kind='dgcn',
                 st_window=6, st_dim=32, st_K=2, st_hidden=32, seg_len=6,
                 tsa_layers=2, n_routers=10, pace=False, corridor=False):
        super().__init__()
        # se guarda con el checkpoint para poder reconstruir el modelo
        self.config = dict(dim=dim, hidden=hidden, dropout=dropout, bank_size=bank_size,
                           n_env=n_env, invariant_ratio=invariant_ratio,
                           n_graph_layers=n_graph_layers, st_kind=st_kind,
                           st_window=st_window, st_dim=st_dim, st_K=st_K,
                           st_hidden=st_hidden, seg_len=seg_len, tsa_layers=tsa_layers,
                           n_routers=n_routers, pace=pace, corridor=corridor)
        self.st_kind = st_kind
        self.pace = pace or corridor

        self.cell_enc = CellEncoder(dim, dropout)
        # el grafo va antes del banco: propagar después difuminaría los prompts
        self.gnn = nn.ModuleList([RegionMessagePassing(dim, dropout)
                                  for _ in range(n_graph_layers)])
        self.npm = NodePromptingModule(dim, bank_size)

        # el estado espaciotemporal depende de la hora, así que va en el contexto
        if st_kind == 'tsa':
            self.stgnn = S.TSAEncoder(dim, seg_len=seg_len, d=dim, out=st_dim,
                                      n_layers=tsa_layers, n_routers=n_routers,
                                      dropout=dropout, K=st_K)
        else:
            self.stgnn = S.UrbanSTGNN(st_window, st_hidden, st_dim, st_K, dropout)

        rep = hidden // 2
        self.route_enc = nn.Sequential(
            nn.Linear(dim * 2 + F_ROUTE, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, rep), nn.LayerNorm(rep))
        self.ctx_enc = nn.Sequential(
            nn.Linear(F_CTX + 2 * st_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, rep), nn.LayerNorm(rep))
        self.gate_route = InvariantGate(rep, invariant_ratio)
        self.gate_ctx = InvariantGate(rep, invariant_ratio)
        self.env_infer = EnvironmentInference(rep * 2, n_env)
        self.head = nn.Sequential(
            nn.Linear(rep, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1))

        self.corridor = (S.CorridorPace(dim, st_dim if st_kind == 'tsa' else 0,
                                        F_ROUTE + F_CTX, d=dim, dropout=dropout)
                         if corridor else None)
        if self.pace:
            self.register_buffer('t_mean', torch.tensor(0.0))
            self.register_buffer('t_std', torch.tensor(1.0))
            self.pace_bias = nn.Parameter(torch.tensor(0.0))

    def set_norms(self, cd):
        """Normalización del tiempo y log-ritmo medio (s/m) de entrenamiento."""
        self.t_mean.fill_(cd.time_log_mean)
        self.t_std.fill_(cd.time_log_std)
        tr = torch.tensor(cd.train_idx)
        log_pace = torch.log(cd.raw_times[tr]) - self._log_dist(cd.feats[tr])
        self.pace_bias.data.fill_(float(log_pace.mean()))

    @staticmethod
    def _log_dist(feats):
        # deshace dist_log = log1p(d) / 10
        return torch.log(torch.expm1(feats[:, I_DIST_LOG] * 10.0).clamp(min=1.0))

    def _resid_feats(self, feats):
        if not self.pace:
            return feats
        f = feats.clone()
        f[:, [I_DIST_NORM, I_DIST_LOG]] = 0.0
        return f

    def encode_cells(self, cd):
        h = self.cell_enc(cd.cell_poi, cd.cell_road, cd.cell_mask, cd.cell_pattern)
        for layer in self.gnn:
            h = layer(h, cd.edge_index, cd.edge_weight)
        return self.npm(h)

    def represent(self, h_cells, o_idx, d_idx, feats, st):
        feats = self._resid_feats(feats)
        psi = self.route_enc(torch.cat([h_cells[o_idx], h_cells[d_idx], feats[:, SL_ROUTE]], 1))
        phi = self.ctx_enc(torch.cat([feats[:, SL_CTX], st], 1))
        return phi, psi

    def forward(self, h_cells, o_idx, d_idx, feats, st, corr=None):
        phi, psi = self.represent(h_cells, o_idx, d_idx, feats, st)
        out = self.head(self.gate_ctx(phi) + self.gate_route(psi)).squeeze(1)
        if not self.pace:
            return out
        logt = self._log_dist(feats) + self.pace_bias
        if self.corridor is not None:
            logt = logt + self.corridor(h_cells, corr['cells'], corr['mask'], corr['pos'],
                                        self._resid_feats(feats), corr.get('st'))
        return (logt - self.t_mean) / self.t_std + out


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

class TSJTTrainer:
    """Entrenamiento conjunto en una etapa (TSJT) con los términos de OOD4TTE.

    Cada paso usa `target_per_batch` viajes de la ciudad objetivo y el resto de
    las fuentes. El gradiente de las fuentes se filtra contra el de la
    objetivo (Target-Skewed Backward) antes de actualizar.
    """

    def __init__(self, model, target, sources, device='cpu', lr_target=2e-4,
                 lr_source=1e-6, alpha=0.5, beta=0.5, env_lr=1e-3, env_gamma=0.1,
                 batch_size=128, target_per_batch=4, lambda_st=0.3,
                 st_mask_ratio=0.3, st_windows=8, eval_chunk=8192):
        """
        alpha, beta    pesos de la varianza del riesgo y del término contrastivo.
        lambda_st      peso de la pérdida de interpolación enmascarada.
        st_mask_ratio  fracción de celdas ocultadas en esa pérdida.
        st_windows     ventanas por paso para esa pérdida (solo 'dgcn').
        """
        self.model = model.to(device)
        self.device = device
        self.target = target
        self.sources = sources
        self.alpha, self.beta = alpha, beta
        self.env_gamma = env_gamma
        self.lr_ratio = lr_source / lr_target
        self.lambda_st = lambda_st
        self.st_mask_ratio = st_mask_ratio
        self.st_windows = st_windows
        self.eval_chunk = eval_chunk
        self.tsa = model.st_kind == 'tsa'

        params = [p for n, p in model.named_parameters() if not n.startswith('env_infer.')]
        self.opt = torch.optim.Adam(params, lr=lr_target, weight_decay=1e-4)
        self.env_opt = torch.optim.Adam(model.env_infer.parameters(), lr=env_lr)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, patience=8, factor=0.5)

        if model.pace:
            model.set_norms(target)
        if not self.tsa:
            self._win = S.window_index(model.stgnn.window).to(device)

        for cd in [target] + sources:
            if model.corridor is not None and cd.corr_cells is None:
                raise ValueError(f"{cd.city}: falta attach_corridors()")
            cd.to(device)

        # sin fuentes, lotes de 4 serían demasiado ruidosos y no activarían la invariancia
        if not sources:
            target_per_batch = batch_size
        n_src = max((batch_size - target_per_batch) // max(len(sources), 1), 1)
        self.tgt_loader = DataLoader(train_dataset(target), batch_size=target_per_batch,
                                     shuffle=True)
        # con drop_last, una fuente más pequeña que el lote dejaría el loader vacío
        self.src_loaders = [DataLoader(train_dataset(s), batch_size=min(n_src, len(s.train_idx)),
                                       shuffle=True, drop_last=True) for s in sources]
        self.history = {k: [] for k in ('train', 'interp', 'val_mae', 'val_rmse', 'val_mape')}

    # estado espaciotemporal

    @staticmethod
    def _st_keys(cd, idx, out_of_fold):
        """Hora de la semana de cada viaje y qué tensor ST lee: el construido
        sin su pliegue (train) o el completo (-1, validación y prueba)."""
        taus = cd.st.how[idx]
        return taus, (cd.st.fold[idx] if out_of_fold else torch.full_like(taus, -1))

    def _window_embed(self, cd, idx, o, d, out_of_fold):
        """DGCN: estado de O y D en la ventana que termina en la hora de salida.

        Los pares (hora, pliegue) se deduplican: un lote toca pocas horas distintas.
        """
        st = cd.st
        taus, src = self._st_keys(cd, idx, out_of_fold)
        uniq, inv = torch.unique(torch.stack([taus, src], 1), dim=0, return_inverse=True)
        emb = torch.empty(len(uniq), cd.n_cells, self.model.stgnn.out_dim, device=self.device)
        for sv in uniq[:, 1].unique().tolist():
            sel = (uniq[:, 1] == sv).nonzero(as_tuple=True)[0]
            H0 = S.assemble_st_input(st.feat, st.spd[sv], st.obs[sv], self._win, uniq[sel, 0])
            emb[sel] = self.model.stgnn(H0, st.P_fwd, st.P_bwd)
        return torch.cat([emb[inv, o], emb[inv, d]], dim=1)

    def _encode_week(self, cd, h, src, node_mask=None):
        st = cd.st
        return self.model.stgnn.encode(st.feat, st.spd[src], st.obs[src], h,
                                       st.P_fwd, st.P_bwd, node_mask)

    def _context(self, cd, h, idx, o, d, out_of_fold):
        """Devuelve el estado ST de O y D [B, 2·st_dim] y el corredor (o None)."""
        corr = None
        if self.model.corridor is not None:
            p = cd.pair_of_trip[idx]
            corr = {'cells': cd.corr_cells[p], 'mask': cd.corr_mask[p], 'pos': cd.corr_pos[p]}
        if not self.tsa:
            return self._window_embed(cd, idx, o, d, out_of_fold), corr

        taus, src = self._st_keys(cd, idx, out_of_fold)
        enc = self.model.stgnn
        st_feat = torch.empty(len(idx), 2 * enc.out_dim, device=self.device)
        if corr is not None:
            corr['st'] = torch.empty(*corr['cells'].shape, enc.out_dim, device=self.device)
        for sv in torch.unique(src).tolist():
            sel = (src == sv).nonzero(as_tuple=True)[0]
            Z = self._encode_week(cd, h, sv)
            t = taus[sel]
            st_feat[sel] = torch.cat([enc.read(Z, t, o[sel]), enc.read(Z, t, d[sel])], 1)
            if corr is not None:
                corr['st'][sel] = enc.read(Z, t, corr['cells'][sel])
        return st_feat, corr

    def _interp_loss(self, cd, h):
        """Interpolación enmascarada (GNNUI): se ocultan celdas completas y se
        reconstruye su velocidad. El error cuenta solo lo oculto y observado."""
        st = cd.st
        spd, obs = st.spd[-1], st.obs[-1]
        C = cd.n_cells
        if self.tsa:
            vis = (torch.rand(C, device=self.device) >= self.st_mask_ratio).float()
            hid = (vis == 0).nonzero(as_tuple=True)[0]
            if len(hid) == 0:
                return torch.zeros((), device=self.device)
            pred = self.model.stgnn.reconstruct(self._encode_week(cd, h, -1, vis), hid)
            target = spd[:, hid].T
            w = obs[:, hid].T
        else:
            taus = torch.randint(0, S.N_HOW, (self.st_windows,), device=self.device)
            M = (torch.rand(self.st_windows, 1, C, 1, device=self.device)
                 >= self.st_mask_ratio).float()
            H0 = S.assemble_st_input(st.feat, spd, obs, self._win, taus, node_mask=M)
            pred = self.model.stgnn.interp(self.model.stgnn(H0, st.P_fwd, st.P_bwd))
            steps = self._win[taus]
            target = spd[steps].permute(0, 2, 1)
            w = (1.0 - M[:, 0]) * obs[steps].permute(0, 2, 1)
        return ((pred - target).abs() * w).sum() / w.sum().clamp(min=1.0)

    # pérdidas

    @staticmethod
    def _concept_risk(env, per):
        return (env * per.unsqueeze(1)).sum(0) / env.sum(0).clamp(min=1e-6)

    @staticmethod
    def _contrastive(phi, psi, env, max_n=256):
        if phi.size(0) > max_n:
            sel = torch.randperm(phi.size(0), device=phi.device)[:max_n]
            phi, psi, env = phi[sel], psi[sel], env[sel]
        perm = torch.randperm(phi.size(0), device=phi.device)
        a_e, a_t = _double_center(env), _double_center(psi)
        return F.softplus(partial_distance_corr(_double_center(phi), a_t, a_e)
                          - partial_distance_corr(_double_center(phi[perm]), a_t, a_e))

    def _loss_on(self, cd, batch):
        feats, o, d, y, idx = (x.to(self.device) for x in batch)
        h = self.model.encode_cells(cd)
        st_feat, corr = self._context(cd, h, idx, o, d, out_of_fold=True)
        per = (self.model(h, o, d, feats, st_feat, corr) - y) ** 2
        loss = per.mean()
        extras = {}
        # la varianza entre entornos necesita un lote mínimo
        if feats.size(0) >= 8:
            phi, psi = self.model.represent(h, o, d, feats, st_feat)
            env = self.model.env_infer(phi, psi).detach()
            rv = self._concept_risk(env, per).var(unbiased=False)
            loss = loss + self.alpha * rv + self.beta * self._contrastive(phi, psi, env)
            extras = {'phi': phi.detach(), 'psi': psi.detach(), 'per': per.detach()}
        li = self._interp_loss(cd, h)
        extras['interp'] = li.item()
        return loss + self.lambda_st * li, extras

    def _grads(self):
        return [None if p.grad is None else p.grad.detach().clone()
                for p in self.model.parameters()]

    def _apply_tsb(self, g_t, g_s):
        """Target-Skewed Backward (TSJT, Ec. 10), por tensor de parámetros.

            g_par = <g_s, g_t> / <g_t, g_t> · g_t
            grad  = g_t + r · (g_s - g_par + 1[<g_s, g_t> > 0] · g_par)

        La parte ortogonal de las fuentes se acepta siempre; la paralela solo si
        no va contra la objetivo. r = lr_source / lr_target, metido en el
        gradiente para usar un solo Adam.
        """
        for p, gt, gs in zip(self.model.parameters(), g_t, g_s):
            if gs is None:
                p.grad = gt
            elif gt is None:
                p.grad = self.lr_ratio * gs
            else:
                dot = (gs * gt).sum()
                g_par = dot / (gt * gt).sum().clamp(min=1e-12) * gt
                keep = 1.0 if dot > 0 else 0.0
                p.grad = gt + self.lr_ratio * (gs - g_par + keep * g_par)

    def _update_env(self, ex):
        """Ajusta la partición en entornos: riesgo desigual entre entornos y
        asignaciones seguras pero balanceadas."""
        self.env_opt.zero_grad()
        env = self.model.env_infer(ex['phi'], ex['psi'])
        risk = self._concept_risk(env, ex['per'])
        marg = env.mean(0)
        h_marg = -(marg * (marg + 1e-8).log()).sum()
        h_cond = -(env * (env + 1e-8).log()).sum(1).mean()
        (risk.var(unbiased=False) - self.env_gamma * h_marg + self.env_gamma * h_cond).backward()
        self.env_opt.step()

    def _step(self, tgt_batch, src_batches):
        self.opt.zero_grad(set_to_none=True)
        loss_t, ex_t = self._loss_on(self.target, tgt_batch)
        loss_t.backward()
        if src_batches:
            g_t = self._grads()
            self.opt.zero_grad(set_to_none=True)
            loss_s = sum(self._loss_on(cd, b)[0] for cd, b in src_batches) / len(src_batches)
            loss_s.backward()
            self._apply_tsb(g_t, self._grads())
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        if 'phi' in ex_t:
            self._update_env(ex_t)
        return loss_t.item(), ex_t

    def _next_source_batches(self, iters):
        out = []
        for k, dl in enumerate(self.src_loaders):
            try:
                b = next(iters[k])
            except StopIteration:
                iters[k] = iter(dl)
                b = next(iters[k])
            out.append((self.sources[k], b))
        return out

    def train(self, epochs=60, patience=12, steps_per_epoch=None):
        best, best_w, bad = float('inf'), None, 0
        src_iters = [iter(dl) for dl in self.src_loaders]

        for ep in range(1, epochs + 1):
            self.model.train()
            losses, interps = [], []
            for step, tb in enumerate(self.tgt_loader):
                if steps_per_epoch and step >= steps_per_epoch:
                    break
                loss, ex = self._step(tb, self._next_source_batches(src_iters))
                losses.append(loss)
                interps.append(ex['interp'])

            mae, rmse, mape = self.evaluate(self.target, self.target.val_idx)
            self.sched.step(mae)
            for k, v in (('train', np.mean(losses)), ('interp', np.mean(interps)),
                         ('val_mae', mae), ('val_rmse', rmse), ('val_mape', mape)):
                self.history[k].append(float(v))

            if mae < best:
                best, bad = mae, 0
                best_w = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
            if ep % 5 == 0:
                print(f"Época {ep:3d} | loss={self.history['train'][-1]:.4f}  "
                      f"MAE={mae:6.1f}s  RMSE={rmse:6.1f}s  MAPE={mape:5.2f}%  "
                      f"interp={self.history['interp'][-1]:.3f}")
            if bad >= patience:
                print(f"Early stopping en época {ep}")
                break

        if best_w:
            self.model.load_state_dict(best_w)
        return self.history

    @torch.no_grad()
    def evaluate(self, cd, idx):
        """MAE y RMSE en segundos, MAPE en %."""
        if not idx:
            return float('nan'), float('nan'), float('nan')
        self.model.eval()
        h = self.model.encode_cells(cd)
        ii = torch.tensor(idx, dtype=torch.long)
        preds = []
        for chunk in ii.split(self.eval_chunk):
            o = cd.o_cell[chunk].to(self.device)
            d = cd.d_cell[chunk].to(self.device)
            st_feat, corr = self._context(cd, h, chunk.to(self.device), o, d, out_of_fold=False)
            preds.append(self.model(h, o, d, cd.feats[chunk].to(self.device), st_feat, corr).cpu())
        secs = torch.exp(torch.cat(preds) * cd.time_log_std + cd.time_log_mean)
        true = cd.raw_times[ii]
        err = secs - true
        return (float(err.abs().mean()),
                float(torch.sqrt((err ** 2).mean())),
                float((err.abs() / true.clamp(min=1)).mean() * 100))

    @torch.no_grad()
    def evaluate_by_hour(self, cd, idx):
        """Métricas por franja de 6 h y el peor grupo, como reporta OOD4TTE."""
        buckets = {}
        hours = (cd.st.how[torch.tensor(idx, device=self.device)] % 24).tolist()
        for i, h in zip(idx, hours):
            buckets.setdefault(h // 6, []).append(i)
        out = {f"{b * 6:02d}-{b * 6 + 5:02d}h": self.evaluate(cd, ids)
               for b, ids in sorted(buckets.items())}
        if out:
            out['peor_grupo'] = max(out.values(), key=lambda m: m[0])
        return out

    def save(self, path):
        torch.save({'model_state': self.model.state_dict(),
                    'config': self.model.config,
                    'norm': {'time_log_mean': self.target.time_log_mean,
                             'time_log_std': self.target.time_log_std},
                    'history': self.history}, path)
        print(f"Modelo guardado -> {path}")
