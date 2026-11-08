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
    # red vial OSM entre las celdas de O y D: medidas relativas, 0 sin red (net_mask)
    'net_circuity', 'net_pace', 'net_mask',
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
    od_dist: torch.Tensor          # [N] distancia O-D en metros
    route_circuity: torch.Tensor   # [N] log(longitud real / distancia O-D): SOLO objetivo auxiliar
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
    # (dist_mean, dist_std, time_log_mean, time_log_std) calculadas con el train de la objetivo
    norms: Optional[tuple] = None

    @property
    def n_cells(self):
        return self.cell_poi.size(0)

    def to(self, device):
        """Mueve lo que se usa por ciudad. Los tensores por viaje (feats,
        o_cell, ...) se quedan en CPU y viajan por lotes."""
        for name in ('cell_poi', 'cell_road', 'cell_mask', 'cell_pattern', 'edge_weight',
                     'route_circuity'):
            setattr(self, name, getattr(self, name).to(device))
        self.edge_index = tuple(e.to(device) for e in self.edge_index)
        self.st = self.st.to(device)
        return self


# ---------------------------------------------------------------------------
# Features y tensores
# ---------------------------------------------------------------------------

def hour_and_dow(ts):
    """Hora del día y día de la semana (lunes = 0; el 1-1-1970 fue jueves).

    ts debe ser la hora local de la ciudad (`local_ts`), no el timestamp UTC.
    """
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
    hour, dow = hour_and_dow(t['local_ts'])
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
        h = hour_and_dow(t['local_ts'])[0]
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

    hd = np.array([hour_and_dow(t['local_ts']) for t in trips], dtype=np.int64)
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


def _route_circuity(trips):
    """log(longitud del recorrido real / distancia en línea recta), recortado a [0, 2].

    Viene de la trayectoria, así que es la etiqueta disfrazada: solo se usa como
    objetivo auxiliar de entrenamiento y nunca como entrada. NaN si la caché no
    trae la longitud del recorrido.
    """
    r = np.array([t.get('route_len', np.nan) / t['od_dist'] for t in trips])
    return np.clip(np.log(np.maximum(r, 1.0)), 0.0, 2.0).astype(np.float32)


def _standardize(x):
    """Z-score de cada feature de celda dentro de la ciudad.

    Una sonda lineal distinguía Porto de Pekín con un 96 % de acierto usando
    solo las features crudas de celda: sus niveles absolutos (mezcla de POIs,
    perfil horario) son una huella de la ciudad. Con z-score cada celda se
    describe respecto de las demás de su ciudad, que es lo que se puede
    transferir. También pone las tres modalidades en la misma escala: el perfil
    horario (24 valores de ~0,04) apenas pesaba frente a POI y red vial.
    Una columna constante (por ejemplo, una modalidad sin datos OSM) queda en 0.
    """
    std = x.std(axis=0)
    return np.where(std > 1e-6, (x - x.mean(axis=0)) / np.maximum(std, 1e-6), 0.0).astype(np.float32)


def build_city_data(city, norms=None, split='chronological', split_kwargs=None,
                    graph_k=8, target_fraction=None):
    """Tensores de una ciudad. target_fraction recorta el train ANTES de
    construir grafo, patrones, tensores ST y normalización, para que el régimen
    few-shot no vea información de los viajes descartados.

    norms=None calcula la normalización con el train de esta ciudad (la objetivo);
    las fuentes reciben las de la objetivo (`cd.norms`).
    """
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
    if norms is None:
        norms = train_norms(city, tr)
    dist_mean, dist_std, time_log_mean, time_log_std = norms

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
        od_dist=torch.tensor([t['od_dist'] for t in city.trips], dtype=torch.float32),
        route_circuity=torch.tensor(_route_circuity(city.trips)),
        cell_poi=torch.tensor(_standardize(poi)), cell_road=torch.tensor(_standardize(road)),
        cell_mask=torch.tensor(mask),
        cell_pattern=torch.tensor(_standardize(_cell_patterns(city, tr))),
        edge_index=(torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)),
        edge_weight=torch.tensor(val, dtype=torch.float32),
        st=build_st_tensors(city, tr, graph_k),
        train_idx=tr, val_idx=va, test_idx=te,
        time_log_mean=time_log_mean, time_log_std=time_log_std, norms=norms)
    return cd


def train_norms(city, train_idx):
    """Media y desviación de la distancia O-D y del log-tiempo, solo con viajes de train.

    Con todos los viajes, la normalización incluía las etiquetas del test y, en
    few-shot, las de los viajes descartados: con 800 viajes de train el modelo
    recibía la duración media de 100.000.
    """
    trips = [city.trips[i] for i in train_idx]
    d = np.array([t['od_dist'] for t in trips])
    lt = np.log(np.maximum([t['travel_time'] for t in trips], 1.0))
    return float(d.mean()), float(d.std() + 1e-8), float(lt.mean()), float(lt.std() + 1e-8)


def metrics(pred, true):
    """MAE, RMSE y MdAE (mediana del error absoluto) en segundos; MAPE y sesgo
    (mediana del error relativo) en %. MdAE no se mueve con unos pocos viajes de
    duración absurda, que en datos sin marca de ocupado dominan MAE y RMSE."""
    if len(true) == 0:
        return dict.fromkeys(('mae', 'rmse', 'mdae', 'mape', 'sesgo'), float('nan'))
    err = pred - true
    rel = err / np.maximum(true, 1.0)
    return {'mae': float(np.abs(err).mean()), 'rmse': float(np.sqrt((err ** 2).mean())),
            'mdae': float(np.median(np.abs(err))),
            'mape': float(np.abs(rel).mean() * 100), 'sesgo': float(np.median(rel) * 100)}


# franjas fijas en km: significan lo mismo en cualquier ciudad, split o
# fracción de datos, y permiten comparar experimentos
DIST_BANDS_KM = [0, 2, 5, 10, 20, None]


def band_label(lo, hi):
    return f"{lo}-{hi} km" if hi is not None else f"> {lo} km"


def report(pred, true, hour, dist):
    """Métricas globales, por franja de 6 h y por franja fija de distancia en km.

    El peor grupo horario se reporta aparte, como en OOD4TTE: un buen promedio
    puede esconder una hora punta mala.
    """
    out = {'total': metrics(pred, true), 'por_hora': {}, 'por_franja_km': {}}
    km = dist / 1000
    for lo, hi in zip(DIST_BANDS_KM[:-1], DIST_BANDS_KM[1:]):
        sel = (km >= lo) & ((km < hi) if hi is not None else True)
        out['por_franja_km'][band_label(lo, hi)] = {**metrics(pred[sel], true[sel]),
                                                     'n': int(sel.sum())}
    for b in np.unique(hour // 6):
        sel = hour // 6 == b
        out['por_hora'][f"{b * 6:02d}-{b * 6 + 5:02d}h"] = metrics(pred[sel], true[sel])
    out['peor_hora'] = max(out['por_hora'].values(), key=lambda m: m['mae'])
    return out


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
    """Compuerta suave por dimensión, z · σ(W z) (OOD4TTE).

    La versión anterior anulaba siempre el 50 % de las dimensiones con menor
    varianza entre muestras, que son las que menos información llevan.
    """

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, z):
        return z * torch.sigmoid(self.proj(z))


def mmd_rbf(x, y):
    """MMD² con una mezcla de núcleos gaussianos (ancho = mediana de distancias x 0,5/1/2)."""
    z = torch.cat([x, y])
    d2 = torch.cdist(z, z).pow(2)
    bw = d2.detach().median().clamp(min=1e-6)
    k = sum(torch.exp(-d2 / (bw * s)) for s in (0.5, 1.0, 2.0))
    n = len(x)
    return k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()


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

    El estado ST sale de UrbanSTGNN (DGCN por ventanas de st_window horas).
    pace      log t = β·log d + b + residuo. β y b se aprenden y arrancan del
              ajuste por mínimos cuadrados del train (β ≈ 0,4-0,5 en Porto: los
              viajes largos son proporcionalmente más rápidos).
    """

    def __init__(self, dim=64, hidden=128, dropout=0.1, bank_size=64, n_env=8,
                 n_graph_layers=2, st_window=6, st_dim=32, st_K=2, st_hidden=32,
                 pace=False):
        super().__init__()
        # se guarda con el checkpoint para poder reconstruir el modelo
        self.config = dict(dim=dim, hidden=hidden, dropout=dropout, bank_size=bank_size,
                           n_env=n_env, n_graph_layers=n_graph_layers,
                           st_window=st_window, st_dim=st_dim, st_K=st_K,
                           st_hidden=st_hidden, pace=pace)
        self.pace = pace

        self.cell_enc = CellEncoder(dim, dropout)
        # el grafo va antes del banco: propagar después difuminaría los prompts
        self.gnn = nn.ModuleList([RegionMessagePassing(dim, dropout)
                                  for _ in range(n_graph_layers)])
        self.npm = NodePromptingModule(dim, bank_size)

        # el estado espaciotemporal depende de la hora, así que va en el contexto
        self.stgnn = S.UrbanSTGNN(st_window, st_hidden, st_dim, st_K, dropout)

        rep = hidden // 2
        # entrada de ruta: h_o, h_d, su producto (interacción O x D) y la geometría
        self.route_enc = nn.Sequential(
            nn.Linear(dim * 3 + F_ROUTE, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, rep), nn.LayerNorm(rep))
        self.ctx_enc = nn.Sequential(
            nn.Linear(F_CTX + 2 * st_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, rep), nn.LayerNorm(rep))
        # objetivo auxiliar: circuidad real del recorrido, predicha desde la ruta
        self.aux_circuity = nn.Linear(rep, 1)
        self.gate_route = InvariantGate(rep)
        self.gate_ctx = InvariantGate(rep)
        self.env_infer = EnvironmentInference(rep * 2, n_env)
        self.head = nn.Sequential(
            nn.Linear(rep, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1))

        if self.pace:
            self.register_buffer('t_mean', torch.tensor(0.0))
            self.register_buffer('t_std', torch.tensor(1.0))
            self.dist_exp = nn.Parameter(torch.tensor(1.0))
            self.pace_bias = nn.Parameter(torch.tensor(0.0))

    def set_norms(self, cd):
        """Normalización del tiempo y ajuste inicial log t = β·log d + b sobre el train."""
        self.t_mean.fill_(cd.time_log_mean)
        self.t_std.fill_(cd.time_log_std)
        tr = torch.tensor(cd.train_idx)
        x, y = self._log_dist(cd.feats[tr]), torch.log(cd.raw_times[tr])
        beta = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
        self.dist_exp.data.fill_(float(beta))
        self.pace_bias.data.fill_(float(y.mean() - beta * x.mean()))

    @staticmethod
    def _log_dist(feats):
        # deshace dist_log = log1p(d) / 10
        return torch.log(torch.expm1(feats[:, I_DIST_LOG] * 10.0).clamp(min=1.0))

    def _resid_feats(self, feats):
        """Con pace, el residuo no ve la distancia: fuera del rango de train
        extrapola mal, y la escala ya la lleva β·log d."""
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
        h_o, h_d = h_cells[o_idx], h_cells[d_idx]
        # el producto h_o * h_d deja expresar qué tiene de particular cada par de
        # celdas sin memorizar el par: se calcula desde embeddings transferibles,
        # a diferencia de un historial por par O-D, que sería una tabla por ciudad
        psi = self.route_enc(torch.cat([h_o, h_d, h_o * h_d, feats[:, SL_ROUTE]], 1))
        phi = self.ctx_enc(torch.cat([feats[:, SL_CTX], st], 1))
        return phi, psi

    def forward(self, h_cells, o_idx, d_idx, feats, st):
        phi, psi = self.represent(h_cells, o_idx, d_idx, feats, st)
        out = self.head(self.gate_ctx(phi) + self.gate_route(psi)).squeeze(1)
        if not self.pace:
            return out
        logt = self.dist_exp * self._log_dist(feats) + self.pace_bias
        return (logt - self.t_mean) / self.t_std + out


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

class TSJTTrainer:
    """Entrenamiento conjunto en una etapa (TSJT) con los términos de OOD4TTE.

    Cada paso usa un lote de la ciudad objetivo y otro del mismo tamaño
    repartido entre las fuentes. El gradiente de las fuentes se filtra contra
    el de la objetivo (Target-Skewed Backward) antes de actualizar.
    """

    def __init__(self, model, target, sources, device='cpu', lr_target=1e-3,
                 source_ratio=1.0, alpha=0.5, beta=0.5, env_lr=1e-3, env_gamma=0.1,
                 batch_size=512, lambda_st=0.3, st_mask_ratio=0.3, st_windows=8,
                 eval_chunk=8192, val_size=20000, weight_decay=1e-4, lambda_align=0.1,
                 lambda_aux=0.1):
        """
        source_ratio   tasa de aprendizaje del paso de las fuentes respecto de la
                       de la objetivo (lr_source / lr_target en TSJT). Con 1 se
                       obtuvo el mejor resultado en few-shot (ver docs/fase_E).
        alpha, beta    pesos de la varianza del riesgo y del término contrastivo.
        lambda_st      peso de la pérdida de interpolación enmascarada.
        st_mask_ratio  fracción de celdas ocultadas en esa pérdida.
        st_windows     ventanas por paso para esa pérdida (solo 'dgcn').
        val_size       viajes de validación evaluados por época (muestra fija).
        lambda_align   peso del término MMD que alinea los embeddings de celda de
                       la objetivo con los de las fuentes (solo con fuentes).
        lambda_aux     peso del objetivo auxiliar de circuidad del recorrido real.
        """
        self.model = model.to(device)
        self.device = device
        self.target = target
        self.sources = sources
        self.alpha, self.beta = alpha, beta
        self.env_gamma = env_gamma
        self.lr = lr_target
        self.lr_ratio = source_ratio
        self.lambda_align = lambda_align
        self.lambda_aux = lambda_aux
        self.lambda_st = lambda_st
        self.st_mask_ratio = st_mask_ratio
        self.st_windows = st_windows
        self.eval_chunk = eval_chunk
        self.batch_size = batch_size
        # muestra fija: la misma en todas las épocas para que el early stopping sea comparable
        val = np.asarray(target.val_idx)
        if len(val) > val_size:
            val = np.sort(np.random.RandomState(0).choice(val, val_size, replace=False))
        self.val_idx = val.tolist()

        params = [p for n, p in model.named_parameters() if not n.startswith('env_infer.')]
        # AdamW: con Adam el decaimiento se sumaba al gradiente y se normalizaba, así
        # que los parámetros con poca señal (banco de prompts, su proyección de
        # consulta, el vector de modalidad ausente) se empujaban a cero a ritmo ~lr
        self.opt = torch.optim.AdamW(params, lr=lr_target, weight_decay=weight_decay)
        self.env_opt = torch.optim.Adam(model.env_infer.parameters(), lr=env_lr)
        # optimizador propio del gradiente filtrado de las fuentes (ver _step)
        self.src_opt = torch.optim.AdamW(params, lr=lr_target * source_ratio, weight_decay=0.0)

        if model.pace:
            model.set_norms(target)
        self._win = S.window_index(model.stgnn.window).to(device)

        for cd in [target] + sources:
            cd.to(device)

        # La objetivo recibe el mismo lote con o sin fuentes, y las fuentes se reparten
        # un lote del mismo tamaño. Así la objetivo ve los mismos datos por época en
        # los dos casos y la diferencia medida es solo el aporte de las fuentes (antes,
        # con 64 viajes de la objetivo por lote, veía 8 veces menos datos).
        n_src = max(batch_size // max(len(sources), 1), 1)
        self.tgt_loader = DataLoader(train_dataset(target),
                                     batch_size=min(batch_size, len(target.train_idx)),
                                     shuffle=True, drop_last=True)
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

    def _interp_loss(self, cd):
        """Interpolación enmascarada (GNNUI): se ocultan celdas completas y se
        reconstruye su velocidad. El error cuenta solo lo oculto y observado."""
        st = cd.st
        spd, obs = st.spd[-1], st.obs[-1]
        C = cd.n_cells
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
        st_feat = self._window_embed(cd, idx, o, d, out_of_fold=True)
        # Huber: cuadrática cerca del objetivo y lineal lejos, para no dejarse
        # arrastrar por viajes anómalos (taxímetro olvidado, desvíos largos)
        per = F.huber_loss(self.model(h, o, d, feats, st_feat), y,
                           reduction='none', delta=1.0)
        loss = per.mean()
        extras = {}
        phi, psi = self.model.represent(h, o, d, feats, st_feat)
        if self.lambda_aux > 0:
            # el recorrido real enseña a la representación de ruta cuánto se desvían
            # los viajes de la recta; en inferencia esta cabeza no se usa
            target = cd.route_circuity[idx]
            ok = torch.isfinite(target)
            if ok.any():
                pred_c = self.model.aux_circuity(psi[ok]).squeeze(1)
                aux = F.huber_loss(pred_c, target[ok], delta=0.5)
                loss = loss + self.lambda_aux * aux
        # la varianza entre entornos necesita un lote mínimo
        if feats.size(0) >= 8:
            env = self.model.env_infer(phi, psi).detach()
            rv = self._concept_risk(env, per).var(unbiased=False)
            loss = loss + self.alpha * rv + self.beta * self._contrastive(phi, psi, env)
            extras = {'phi': phi.detach(), 'psi': psi.detach(), 'per': per.detach()}
        li = self._interp_loss(cd)
        extras['interp'] = li.item()
        return loss + self.lambda_st * li, extras

    def _grads(self):
        return [None if p.grad is None else p.grad.detach().clone()
                for p in self.model.parameters()]

    @staticmethod
    def _filter_source_grads(g_t, g_s):
        """Target-Skewed Backward (TSJT, Ec. 10), por tensor de parámetros.

            g_par = <g_s, g_t> / <g_t, g_t> · g_t
            filtro = g_s - g_par + 1[<g_s, g_t> > 0] · g_par

        La parte ortogonal de las fuentes se acepta siempre; la paralela solo si
        no va contra la objetivo.
        """
        out = []
        for gt, gs in zip(g_t, g_s):
            if gs is None or gt is None:
                out.append(gs)
                continue
            dot = (gs * gt).sum()
            g_par = dot / (gt * gt).sum().clamp(min=1e-12) * gt
            # (dot > 0) como tensor: un `if` forzaría una sincronización con la GPU por parámetro
            keep = (dot > 0).to(gt.dtype)
            out.append(gs - g_par + keep * g_par)
        return out

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

    def _align_loss(self, max_cells=512):
        """MMD entre los embeddings de celda de la objetivo y los de cada fuente.

        La sonda de ciudad acertaba el 100 % con el embedding final: nada en el
        modelo acercaba las celdas de ciudades distintas. Este término empuja las
        dos distribuciones a solaparse, de modo que una celda de Porto y una de
        Pekín con contexto parecido caigan en la misma zona del espacio.
        """
        h_t = self.model.encode_cells(self.target)
        h_t = h_t[torch.randperm(len(h_t), device=self.device)[:max_cells]]
        total = 0.0
        for cd in self.sources:
            h_s = self.model.encode_cells(cd)
            h_s = h_s[torch.randperm(len(h_s), device=self.device)[:max_cells]]
            total = total + mmd_rbf(h_t, h_s)
        return total / len(self.sources)

    def _step(self, tgt_batch, src_batches):
        self.opt.zero_grad(set_to_none=True)
        loss_t, ex_t = self._loss_on(self.target, tgt_batch)
        if src_batches and self.lambda_align > 0:
            # entra en el gradiente de la objetivo para que el filtro TSB no lo atenúe
            loss_t = loss_t + self.lambda_align * self._align_loss()
        loss_t.backward()
        g_src = None
        if src_batches:
            g_t = self._grads()
            self.opt.zero_grad(set_to_none=True)
            # un backward por fuente: el gradiente se acumula igual que con la
            # pérdida media, pero solo hay un grafo de cálculo vivo en la GPU
            for cd, b in src_batches:
                (self._loss_on(cd, b)[0] / len(src_batches)).backward()
            g_src = self._filter_source_grads(g_t, self._grads())
            for p, g in zip(self.model.parameters(), g_t):
                p.grad = g
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        if g_src is not None:
            # Paso propio de las fuentes, con tasa lr_source = source_ratio x lr actual,
            # como las dos tasas de TSJT. Antes se sumaba r·g_fuente al gradiente de la
            # objetivo en un solo Adam, pero Adam normaliza el paso por parámetro: donde
            # el gradiente de la objetivo era chico, el término de las fuentes movía el
            # parámetro casi a tasa completa aunque r fuera 0,005, y el entrenamiento se
            # volvía inestable.
            for p, g in zip(self.model.parameters(), g_src):
                p.grad = g
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            for group in self.src_opt.param_groups:
                group['lr'] = self.lr_ratio * self.opt.param_groups[0]['lr']
            self.src_opt.step()
        self.sched.step()
        if 'phi' in ex_t:
            self._update_env(ex_t)
        return loss_t.item(), ex_t

    @staticmethod
    def _next(loader, iters, k):
        """Siguiente lote del loader k, reiniciándolo al agotarse."""
        try:
            return next(iters[k])
        except StopIteration:
            iters[k] = iter(loader)
            return next(iters[k])

    def train(self, epochs=30, patience=6, steps_per_epoch=None):
        best, best_w, bad = float('inf'), None, 0
        loaders = [self.tgt_loader] + self.src_loaders
        iters = [iter(dl) for dl in loaders]
        # una época = una pasada por el train de la objetivo, con o sin fuentes, así
        # que ambos casos hacen los mismos pasos. Un mínimo de pasos por época
        # (probado: 50) hacía sobreajustar la objetivo en few-shot desde la época 3
        n_steps = math.ceil(len(self.target.train_idx) / self.batch_size)
        if steps_per_epoch:
            n_steps = min(n_steps, steps_per_epoch)
        # calentamiento corto y decaimiento coseno a lo largo de todo el entrenamiento
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            self.opt, max_lr=self.lr, total_steps=epochs * n_steps, pct_start=0.05,
            cycle_momentum=False)

        for ep in range(1, epochs + 1):
            self.model.train()
            losses, interps = [], []
            for _ in range(n_steps):
                tb = self._next(loaders[0], iters, 0)
                sb = [(s, self._next(dl, iters, k + 1))
                      for k, (s, dl) in enumerate(zip(self.sources, self.src_loaders))]
                loss, ex = self._step(tb, sb)
                losses.append(loss)
                interps.append(ex['interp'])

            val = self.evaluate(self.target, self.val_idx)
            mae = val['mae']
            for k, v in (('train', np.mean(losses)), ('interp', np.mean(interps)),
                         ('val_mae', mae), ('val_rmse', val['rmse']), ('val_mape', val['mape'])):
                self.history[k].append(float(v))

            if mae < best:
                best, bad = mae, 0
                best_w = {k: v.clone() for k, v in self.model.state_dict().items()}
            elif ep > epochs // 2:
                # la paciencia cuenta solo en la segunda mitad: antes la tasa de
                # aprendizaje sigue alta y la validación oscila sin haber convergido
                bad += 1
            if ep % 5 == 0:
                print(f"Época {ep:3d} | loss={self.history['train'][-1]:.4f}  "
                      f"MAE={mae:6.1f}s  RMSE={val['rmse']:6.1f}s  MAPE={val['mape']:5.2f}%  "
                      f"interp={self.history['interp'][-1]:.3f}")
            if bad >= patience:
                print(f"Early stopping en época {ep}")
                break

        if best_w:
            self.model.load_state_dict(best_w)
        return self.history

    @torch.no_grad()
    def predict(self, cd, idx):
        """Tiempo de viaje predicho en segundos (numpy) para los viajes idx."""
        self.model.eval()
        h = self.model.encode_cells(cd)
        preds = []
        for chunk in torch.tensor(idx, dtype=torch.long).split(self.eval_chunk):
            o = cd.o_cell[chunk].to(self.device)
            d = cd.d_cell[chunk].to(self.device)
            st_feat = self._window_embed(cd, chunk.to(self.device), o, d, out_of_fold=False)
            preds.append(self.model(h, o, d, cd.feats[chunk].to(self.device), st_feat).cpu())
        if not preds:
            return np.zeros(0)
        return torch.exp(torch.cat(preds) * cd.time_log_std + cd.time_log_mean).numpy()

    def evaluate(self, cd, idx):
        return metrics(self.predict(cd, idx), cd.raw_times[idx].numpy())

    def save(self, path):
        torch.save({'model_state': self.model.state_dict(),
                    'config': self.model.config,
                    'norm': {'time_log_mean': self.target.time_log_mean,
                             'time_log_std': self.target.time_log_std},
                    'history': self.history}, path)
        print(f"Modelo guardado -> {path}")
