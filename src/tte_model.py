"""
tte_model.py
============
Reemplaza a gnn_travel_time.py. Estimación de tiempo de viaje origen-destino
(OD TTE) sobre trayectorias GPS, sin GTFS y sin grafo de servicio.

Qué aporta cada referencia
--------------------------
TransferTraj (NeurIPS 2025)
    Representación por MODALIDADES (espacial, temporal, POI, red vial) y
    codificación región-agnóstica. Aquí las modalidades espaciales son
    relativas (radiales y diferenciales), nunca coordenadas absolutas: es lo
    que permite que un modelo entrenado en una ciudad se aplique a otra.

OOD4TTE (KDD 2026)
    Separación explícita entre RUTA (T) y CONTEXTO DE TRÁFICO (C), inferencia
    de entornos latentes, bloqueo selectivo por contraste y compuerta
    invariante. Ataca el desplazamiento de distribución DENTRO de una ciudad
    (hora punta, ruta larga vs corta).

TSJT (AAAI 2025)
    Target-Skewed Backward: entrenamiento conjunto en una sola etapa sobre
    ciudades fuente y objetivo, filtrando el gradiente de las fuentes para
    conservar solo la parte que no perjudica a la objetivo. Node Prompting
    Module: banco de prompts compartido entre regiones con patrones
    espaciotemporales similares. Ataca el desplazamiento ENTRE ciudades.

TSB sustituye al esquema anterior de preentrenar y luego hacer fine-tuning.
El argumento de TSJT es que el enfoque en dos etapas absorbe ruido de las
fuentes que después el fine-tuning tiene que deshacer; entrenar en una sola
etapa con el gradiente filtrado evita adquirir ese ruido de entrada.

Dependencias: torch + numpy.
"""

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import compat
import trajectory_loader as tl
import urban_context


# ─────────────────────────────────────────────────────────────────────────────
# ESQUEMA DE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

# RUTA (T): geometría del par O-D y del corredor recto entre ambos.
# NO incluye nada derivado del trayecto realmente recorrido: en este dataset el
# tiempo de viaje es (n_puntos-1)*15 s, así que la polilínea ES la etiqueta.
ROUTE_NAMES = [
    'dist_norm',            # distancia O-D normalizada globalmente
    'dist_log',             # log-distancia (el tiempo escala de forma no lineal)
    'bearing_sin', 'bearing_cos',
    'o_radial',             # distancia de O al centro urbano / radio  ∈[0,1]
    'd_radial',
    'radial_delta',         # d_radial - o_radial: entrar o salir del centro
    'inter_per_km', 'turn_per_km',
    'corr_arterial', 'corr_colectora', 'corr_local', 'corr_peatonal',
]

# CONTEXTO (C): cuándo y en qué condiciones se hace el viaje.
CTX_NAMES = [
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend',
    'call_A', 'call_B', 'call_C',
    'day_A', 'day_B', 'day_C',
    'signal_per_km', 'friction', 'poi_corridor_q', 'ctx_mask',
]

F_ROUTE = len(ROUTE_NAMES)          # 13
F_CTX = len(CTX_NAMES)              # 15

# Modalidades de la CELDA (región), idénticas a las que antes tenía la parada.
CELL_POI_NAMES = list(urban_context.NODE_POI_NAMES)
CELL_ROAD_NAMES = list(urban_context.NODE_ROAD_NAMES)
F_CELL_POI = len(CELL_POI_NAMES)
F_CELL_ROAD = len(CELL_ROAD_NAMES)
N_HOURS = 24                        # perfil horario de demanda por celda

SL_ROUTE = slice(0, F_ROUTE)
SL_CTX = slice(F_ROUTE, F_ROUTE + F_CTX)


@dataclass
class CityData:
    """Tensores de una ciudad, listos para el entrenamiento."""
    city: str
    city_id: int
    feats: torch.Tensor            # [N, F_ROUTE + F_CTX]
    o_cell: torch.Tensor           # [N]
    d_cell: torch.Tensor           # [N]
    times: torch.Tensor            # [N]  log-tiempo normalizado
    raw_times: torch.Tensor        # [N]  segundos
    cell_poi: torch.Tensor         # [C, F_CELL_POI]
    cell_road: torch.Tensor        # [C, F_CELL_ROAD]
    cell_mask: torch.Tensor        # [C, 2]  disponibilidad POI / vial
    cell_pattern: torch.Tensor     # [C, N_HOURS + 2]  patrón espaciotemporal
    train_idx: List[int] = field(default_factory=list)
    val_idx: List[int] = field(default_factory=list)
    test_idx: List[int] = field(default_factory=list)
    time_log_mean: float = 0.0
    time_log_std: float = 1.0

    @property
    def n_cells(self):
        return self.cell_poi.size(0)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE TENSORES
# ─────────────────────────────────────────────────────────────────────────────

def _trip_vector(t, city, dist_mean, dist_std):
    """Vector [RUTA | CONTEXTO] de un viaje. Sin nada derivado del recorrido."""
    d = t['od_dist']
    lat_c, lon_c, R = city.lat_c, city.lon_c, city.radius
    o_rad = min(tl._haversine(t['o_lat'], t['o_lon'], lat_c, lon_c) / R, 2.0)
    d_rad = min(tl._haversine(t['d_lat'], t['d_lon'], lat_c, lon_c) / R, 2.0)
    br = math.radians(tl._bearing(t['o_lat'], t['o_lon'], t['d_lat'], t['d_lon']))

    route = [
        (d - dist_mean) / dist_std,
        math.log1p(d) / 10.0,
        math.sin(br), math.cos(br),
        o_rad, d_rad, d_rad - o_rad,
        float(t.get('inter_per_km', 0.0)),
        float(t.get('turn_per_km', 0.0)),
        float(t.get('corr_arterial', 0.0)),
        float(t.get('corr_colectora', 0.0)),
        float(t.get('corr_local', 0.0)),
        float(t.get('corr_peatonal', 0.0)),
    ]

    # el TIMESTAMP es UTC en segundos; hora local y día de la semana
    ts = t['timestamp']
    hour = (ts // 3600) % 24
    dow = ((ts // 86400) + 4) % 7          # 1970-01-01 fue jueves
    call = [1.0 if t['call_type'] == c else 0.0 for c in tl.CALL_TYPES]
    day = [1.0 if t['day_type'] == c else 0.0 for c in tl.DAY_TYPES]

    ctx = [
        math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * dow / 7), math.cos(2 * math.pi * dow / 7),
        1.0 if dow >= 5 else 0.0,
    ] + call + day + [
        float(t.get('signal_per_km', 0.0)),
        float(t.get('friction', 0.0)),
        float(t.get('poi_corridor_q', 0.0)),
        1.0 if t.get('has_ctx') else 0.0,
    ]
    return route + ctx


def _cell_patterns(city, train_idx):
    """
    Patrón espaciotemporal de cada celda: perfil horario de demanda + intensidad
    de origen/destino. Es la consulta (Query) del NPM.

    Se calcula SOLO con los viajes de entrenamiento. Usar el conjunto completo
    metería información del test en los prompts, que es una fuga sutil y difícil
    de detectar después.
    """
    C = city.grid.n_cells
    prof = np.zeros((C, N_HOURS), dtype=np.float64)
    n_o = np.zeros(C)
    n_d = np.zeros(C)
    for i in train_idx:
        t = city.trips[i]
        h = int((t['timestamp'] // 3600) % 24)
        prof[t['o_cell'], h] += 1
        prof[t['d_cell'], h] += 1
        n_o[t['o_cell']] += 1
        n_d[t['d_cell']] += 1
    prof = prof / np.maximum(prof.sum(axis=1, keepdims=True), 1.0)
    # intensidades como rango percentil: comparables entre ciudades
    return np.concatenate([prof,
                           urban_context._quantile_rank(n_o)[:, None],
                           urban_context._quantile_rank(n_d)[:, None]],
                          axis=1).astype(np.float32)


def build_city_data(city: tl.CityTrips, city_id: int, dist_mean, dist_std,
                    time_log_mean, time_log_std, split='chronological',
                    split_kwargs=None) -> CityData:
    split_kwargs = split_kwargs or {}
    if split == 'chronological':
        tr, va, te = city.chronological_split(**split_kwargs)
    elif split == 'ood_hour':
        tr, te = city.ood_split_by_hour(**split_kwargs)
        cut = int(len(tr) * 0.9)
        tr, va = tr[:cut], tr[cut:]
    elif split == 'ood_distance':
        tr, te = city.ood_split_by_distance(**split_kwargs)
        cut = int(len(tr) * 0.9)
        tr, va = tr[:cut], tr[cut:]
    else:
        raise ValueError(f"split desconocido: {split!r}")

    feats = np.array([_trip_vector(t, city, dist_mean, dist_std)
                      for t in city.trips], dtype=np.float32)
    raw = np.array([t['travel_time'] for t in city.trips], dtype=np.float32)
    times = (np.log(np.maximum(raw, 1.0)) - time_log_mean) / time_log_std

    cells = city.cells
    poi = np.array([[c.get(n, 0.0) for n in CELL_POI_NAMES] for c in cells],
                   dtype=np.float32)
    road = np.array([[c.get(n, 0.0) for n in CELL_ROAD_NAMES] for c in cells],
                    dtype=np.float32)
    mask = np.array([[1.0 if c.get('has_poi_ctx') else 0.0,
                      1.0 if c.get('has_road_ctx') else 0.0] for c in cells],
                    dtype=np.float32)

    return CityData(
        city=city.city, city_id=city_id,
        feats=torch.tensor(feats),
        o_cell=torch.tensor([t['o_cell'] for t in city.trips], dtype=torch.long),
        d_cell=torch.tensor([t['d_cell'] for t in city.trips], dtype=torch.long),
        times=torch.tensor(times, dtype=torch.float32),
        raw_times=torch.tensor(raw),
        cell_poi=torch.tensor(poi), cell_road=torch.tensor(road),
        cell_mask=torch.tensor(mask),
        cell_pattern=torch.tensor(_cell_patterns(city, tr)),
        train_idx=tr, val_idx=va, test_idx=te,
        time_log_mean=time_log_mean, time_log_std=time_log_std)


def global_norms(cities: List[tl.CityTrips]):
    """Normalización compartida entre ciudades, para que las escalas sean comparables."""
    d = np.array([t['od_dist'] for c in cities for t in c.trips])
    lt = np.log(np.maximum([t['travel_time'] for c in cities for t in c.trips], 1.0))
    return (float(d.mean()), float(d.std() + 1e-8),
            float(lt.mean()), float(lt.std() + 1e-8))


class TripDataset(Dataset):
    def __init__(self, data: CityData, idx):
        self.idx = torch.tensor(idx, dtype=torch.long)
        self.data = data

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        k = self.idx[i]
        return (self.data.feats[k], self.data.o_cell[k], self.data.d_cell[k],
                self.data.times[k])


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULOS
# ─────────────────────────────────────────────────────────────────────────────

class CellEncoder(nn.Module):
    """
    Codificador de región por modalidades, con token de modalidad ausente.

    Es el Trajectory Modality Mixing de TransferTraj adaptado a celdas: cada
    modalidad tiene su codificador, las ausentes se sustituyen por un embedding
    aprendido, y la fusión es una atención sobre los tokens. Sin el token de
    ausencia, una ciudad sin cobertura OSM entraría con ceros, que el modelo
    leería como "región sin nada alrededor" en vez de como "sin datos".
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.enc_poi = nn.Sequential(nn.Linear(F_CELL_POI, dim), nn.ReLU(),
                                     nn.LayerNorm(dim))
        self.enc_road = nn.Sequential(nn.Linear(F_CELL_ROAD, dim), nn.ReLU(),
                                      nn.LayerNorm(dim))
        self.enc_pat = nn.Sequential(nn.Linear(N_HOURS + 2, dim), nn.ReLU(),
                                     nn.LayerNorm(dim))
        self.missing = nn.Parameter(torch.randn(2, dim) * 0.02)
        self.attn = nn.Linear(dim, 1)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, dim),
                                 nn.LayerNorm(dim))

    def forward(self, poi, road, mask, pattern):
        e_pat = self.enc_pat(pattern)
        e_poi = torch.where(mask[:, :1] > 0.5, self.enc_poi(poi),
                            self.missing[0].expand_as(e_pat))
        e_road = torch.where(mask[:, 1:2] > 0.5, self.enc_road(road),
                             self.missing[1].expand_as(e_pat))
        tok = torch.stack([e_pat, e_poi, e_road], dim=1)
        a = torch.softmax(self.attn(tok), dim=1)
        return self.out((tok * a).sum(1))


class NodePromptingModule(nn.Module):
    """
    Node Prompting Module (TSJT).

    Mantiene un banco de prompts compartido por TODAS las ciudades. Cada celda
    consulta el banco por atención usando su patrón espaciotemporal como Query.
    Dos celdas de ciudades distintas con perfiles de demanda parecidos —una zona
    residencial de Porto y una de Lima— recuperan prompts parecidos, y el modelo
    produce gradientes parecidos sobre ellas. Eso es justamente lo que mejora la
    calidad del gradiente que TSB después filtra.

    Tamaño del banco: la mitad del número de celdas, como recomienda el paper.
    """

    def __init__(self, dim, bank_size=64):
        super().__init__()
        self.bank = nn.Parameter(torch.randn(bank_size, dim) * 0.02)
        self.q = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, h_cell):
        q = self.q(h_cell)
        att = torch.softmax(q @ self.bank.T * self.scale, dim=-1)
        return h_cell + att @ self.bank


class EnvironmentInference(nn.Module):
    """Etiquetas blandas de entorno sobre V conceptos de tráfico (OOD4TTE Eq. 3-5)."""

    def __init__(self, in_dim, n_env=8, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_env))

    def forward(self, phi, psi):
        return torch.softmax(self.net(torch.cat([phi, psi], dim=1)), dim=1)


class InvariantGate(nn.Module):
    """
    Compuerta invariante (OOD4TTE Eq. 11-12): conserva el k% de dimensiones más
    estables. La varianza se acumula en un buffer EMA para que en inferencia la
    máscara sea determinista; con la varianza del lote, la misma consulta daría
    resultados distintos según con quién compartiera batch.
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


def _inner(a, b):
    return (a * b).mean()


def partial_distance_corr(a_c, a_t, a_e, eps=1e-8):
    """Correlación de distancia parcial entre C y T dado E (Székely & Rizzo)."""
    de = _inner(a_e, a_e) + eps
    p_c = a_c - (_inner(a_c, a_e) / de) * a_e
    p_t = a_t - (_inner(a_t, a_e) / de) * a_e
    num = _inner(p_c, p_t).abs()
    den = _inner(p_c, p_c).clamp(min=eps).sqrt() * _inner(p_t, p_t).clamp(min=eps).sqrt()
    return num / (den + eps)


class TravelTimeModel(nn.Module):
    """
    Modelo OD TTE completo.

        celdas O y D --> CellEncoder --> NPM --> h_o, h_d
        [h_o | h_d | RUTA]  --> Ψ1     (representación de ruta)
        CONTEXTO            --> Φ1     (representación de contexto)
        compuertas invariantes --> Φ, Ψ
        cabeza MLP(Φ + Ψ)      --> log-tiempo normalizado
    """

    def __init__(self, dim=64, hidden=128, dropout=0.1, bank_size=64,
                 invariant=True, n_env=8, invariant_ratio=0.5, use_npm=True):
        super().__init__()
        self.config = dict(dim=dim, hidden=hidden, dropout=dropout,
                           bank_size=bank_size, invariant=invariant,
                           n_env=n_env, invariant_ratio=invariant_ratio,
                           use_npm=use_npm)
        self.invariant = invariant
        self.use_npm = use_npm
        self.cell_enc = CellEncoder(dim, dropout)
        self.npm = NodePromptingModule(dim, bank_size) if use_npm else None

        rep = hidden // 2
        self.route_enc = nn.Sequential(
            nn.Linear(dim * 2 + F_ROUTE, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, rep), nn.LayerNorm(rep))
        self.ctx_enc = nn.Sequential(
            nn.Linear(F_CTX, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, rep), nn.LayerNorm(rep))
        if invariant:
            self.gate_route = InvariantGate(rep, invariant_ratio)
            self.gate_ctx = InvariantGate(rep, invariant_ratio)
            self.env_infer = EnvironmentInference(rep * 2, n_env)
        self.head = nn.Sequential(
            nn.Linear(rep, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1))

    def encode_cells(self, poi, road, mask, pattern):
        h = self.cell_enc(poi, road, mask, pattern)
        return self.npm(h) if self.npm is not None else h

    def represent(self, h_cells, o_idx, d_idx, feats):
        psi1 = self.route_enc(torch.cat(
            [h_cells[o_idx], h_cells[d_idx], feats[:, SL_ROUTE]], dim=1))
        phi1 = self.ctx_enc(feats[:, SL_CTX])
        return phi1, psi1

    def forward(self, h_cells, o_idx, d_idx, feats):
        phi1, psi1 = self.represent(h_cells, o_idx, d_idx, feats)
        if self.invariant:
            phi1, psi1 = self.gate_ctx(phi1), self.gate_route(psi1)
        return self.head(phi1 + psi1).squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

class TSJTTrainer:
    """
    Entrenamiento conjunto en una sola etapa (TSJT) con los términos de
    invariancia de OOD4TTE.

    Cada paso procesa un lote mixto: `target_per_batch` muestras de la ciudad
    objetivo y el resto de las fuentes (el paper usa 4 de 128). Se calculan los
    dos gradientes por separado, se descompone el de las fuentes respecto al de
    la objetivo y se recombina según la Eq. 10.
    """

    def __init__(self, model, target: CityData, sources: List[CityData],
                 device='cpu', lr_target=2e-4, lr_source=1e-6,
                 alpha=0.5, beta=0.5, env_lr=1e-3, env_gamma=0.1,
                 batch_size=128, target_per_batch=4, use_tsb=True):
        self.model = model.to(device)
        self.device = device
        self.target = target
        self.sources = sources
        self.alpha, self.beta = alpha, beta
        self.env_gamma = env_gamma
        self.use_tsb = use_tsb and bool(sources)
        self.lr_ratio = lr_source / max(lr_target, 1e-12)
        self.invariant = model.invariant

        params = [p for n, p in model.named_parameters()
                  if not n.startswith('env_infer.')]
        self.opt = torch.optim.Adam(params, lr=lr_target, weight_decay=1e-4)
        self.env_opt = (torch.optim.Adam(model.env_infer.parameters(), lr=env_lr)
                        if self.invariant else None)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, patience=8, factor=0.5)

        self._cells = {}
        for cd in [target] + list(sources):
            self._cells[cd.city_id] = tuple(
                x.to(device) for x in (cd.cell_poi, cd.cell_road,
                                       cd.cell_mask, cd.cell_pattern))

        n_src = max(batch_size - target_per_batch, 1)
        self.tgt_loader = DataLoader(TripDataset(target, target.train_idx),
                                     batch_size=target_per_batch, shuffle=True,
                                     drop_last=False)
        self.src_loaders = [
            DataLoader(TripDataset(s, s.train_idx),
                       batch_size=max(n_src // max(len(sources), 1), 1),
                       shuffle=True, drop_last=True)
            for s in sources]
        self.history = {'train': [], 'val_mae': [], 'val_rmse': [],
                        'val_mape': [], 'risk_var': [], 'con': []}

    # ── pérdidas ─────────────────────────────────────────────────────────────

    def _loss_on(self, cd: CityData, batch):
        feats, o, d, y = (x.to(self.device) for x in batch)
        h = self.model.encode_cells(*self._cells[cd.city_id])
        pred = self.model(h, o, d, feats)
        per = (pred - y) ** 2
        loss = per.mean()
        extras = {}
        if self.invariant and feats.size(0) >= 8:
            phi1, psi1 = self.model.represent(h, o, d, feats)
            env = self.model.env_infer(phi1, psi1)
            risk = self._concept_risk(env.detach(), per)
            rv = risk.var(unbiased=False)
            con = self._contrastive(phi1, psi1, env.detach())
            loss = loss + self.alpha * rv + self.beta * con
            extras = {'risk_var': float(rv.detach()), 'con': float(con.detach()),
                      'phi1': phi1.detach(), 'psi1': psi1.detach(),
                      'per': per.detach()}
        return loss, extras

    @staticmethod
    def _concept_risk(env, per):
        return (env * per.unsqueeze(1)).sum(0) / env.sum(0).clamp(min=1e-6)

    @staticmethod
    def _contrastive(phi1, psi1, env, max_n=256):
        n = phi1.size(0)
        if n > max_n:
            sel = torch.randperm(n, device=phi1.device)[:max_n]
            phi1, psi1, env = phi1[sel], psi1[sel], env[sel]
        perm = torch.randperm(phi1.size(0), device=phi1.device)
        a_e, a_t = _double_center(env), _double_center(psi1)
        return F.softplus(partial_distance_corr(_double_center(phi1), a_t, a_e)
                          - partial_distance_corr(_double_center(phi1[perm]), a_t, a_e))

    # ── Target-Skewed Backward ───────────────────────────────────────────────

    def _grads(self):
        return [p.grad.detach().clone() if p.grad is not None else None
                for p in self.model.parameters()]

    def _apply_tsb(self, g_t, g_s):
        """
        Recombinación de la Eq. 10 de TSJT, por tensor de parámetros:

            θ̇_ST = <θ̇_S, θ̇_T>_F / <θ̇_T, θ̇_T>_F · θ̇_T      (componente paralela)
            θ̇_SV = θ̇_S − θ̇_ST                                (componente ortogonal)
            grad  = θ̇_T + r · (θ̇_SV + 1[<θ̇_ST, θ̇_T> > 0] · θ̇_ST)

        La componente ortogonal siempre se acepta: reduce la pérdida en las
        fuentes sin mover la de la objetivo, es decir, desplaza el espacio de
        búsqueda sin coste. La paralela solo se acepta si apunta en el mismo
        sentido que el gradiente de la objetivo; si va en contra, se descarta.

        r = lr_source / lr_target. El paper aplica dos tasas distintas; aquí se
        codifica la razón en el propio gradiente para poder usar un único
        optimizador Adam con estado consistente.
        """
        for p, gt, gs in zip(self.model.parameters(), g_t, g_s):
            if gt is None and gs is None:
                continue
            if gt is None:
                p.grad = self.lr_ratio * gs
                continue
            if gs is None:
                p.grad = gt
                continue
            dot = (gs * gt).sum()
            nrm = (gt * gt).sum().clamp(min=1e-12)
            g_st = (dot / nrm) * gt
            g_sv = gs - g_st
            keep = 1.0 if dot > 0 else 0.0
            p.grad = gt + self.lr_ratio * (g_sv + keep * g_st)

    def _step(self, tgt_batch, src_batches):
        self.opt.zero_grad(set_to_none=True)
        loss_t, ex_t = self._loss_on(self.target, tgt_batch)
        loss_t.backward()
        g_t = self._grads()

        g_s = None
        loss_s_val = 0.0
        if src_batches:
            self.opt.zero_grad(set_to_none=True)
            loss_s = 0.0
            for cd, b in src_batches:
                ls, _ = self._loss_on(cd, b)
                loss_s = loss_s + ls
            loss_s = loss_s / len(src_batches)
            loss_s.backward()
            g_s = self._grads()
            loss_s_val = float(loss_s.detach())

        if g_s is not None and self.use_tsb:
            self._apply_tsb(g_t, g_s)
        elif g_s is not None:
            # ablación: entrenamiento conjunto ingenuo, sin filtrar el gradiente
            for p, gt, gs in zip(self.model.parameters(), g_t, g_s):
                if gt is None and gs is None:
                    continue
                p.grad = (gt if gt is not None else 0) + \
                         self.lr_ratio * (gs if gs is not None else 0)
        else:
            for p, gt in zip(self.model.parameters(), g_t):
                p.grad = gt

        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

        if self.invariant and ex_t:
            self.env_opt.zero_grad()
            env2 = self.model.env_infer(ex_t['phi1'], ex_t['psi1'])
            r2 = self._concept_risk(env2, ex_t['per'])
            marg = env2.mean(0)
            h_marg = -(marg * (marg + 1e-8).log()).sum()
            h_cond = -(env2 * (env2 + 1e-8).log()).sum(1).mean()
            (r2.var(unbiased=False) - self.env_gamma * h_marg
             + self.env_gamma * h_cond).backward()
            self.env_opt.step()

        return float(loss_t.detach()), loss_s_val, ex_t

    # ── bucle ────────────────────────────────────────────────────────────────

    def train(self, epochs=60, patience=12, verbose=True, checkpoint_path=None,
              steps_per_epoch=None):
        """
        steps_per_epoch acota los pasos por época. Sin él, el número de pasos lo
        fija el tamaño del conjunto objetivo dividido por `target_per_batch`
        (4 en TSJT), lo que con un objetivo grande da miles de pasos por época
        y hace que 'época' deje de ser una unidad comparable entre variantes de
        la ablación.
        """
        best, best_w, bad = float('inf'), None, 0
        src_iters = [iter(dl) for dl in self.src_loaders]

        for ep in range(1, epochs + 1):
            self.model.train()
            tot, rv, cn, nb = 0.0, 0.0, 0.0, 0
            for step, tb in enumerate(self.tgt_loader):
                if steps_per_epoch and step >= steps_per_epoch:
                    break
                sb = []
                for k, dl in enumerate(self.src_loaders):
                    try:
                        b = next(src_iters[k])
                    except StopIteration:
                        src_iters[k] = iter(dl)
                        b = next(src_iters[k])
                    sb.append((self.sources[k], b))
                lt, _, ex = self._step(tb, sb)
                tot += lt
                rv += ex.get('risk_var', 0.0)
                cn += ex.get('con', 0.0)
                nb += 1

            mae, rmse, mape = self.evaluate(self.target, self.target.val_idx)
            self.sched.step(mae)
            self.history['train'].append(tot / max(nb, 1))
            self.history['val_mae'].append(mae)
            self.history['val_rmse'].append(rmse)
            self.history['val_mape'].append(mape)
            self.history['risk_var'].append(rv / max(nb, 1))
            self.history['con'].append(cn / max(nb, 1))

            if mae < best:
                best, bad = mae, 0
                best_w = {k: v.clone() for k, v in self.model.state_dict().items()}
                if checkpoint_path:
                    self.save(checkpoint_path, quiet=True)
            else:
                bad += 1
            if verbose and ep % 5 == 0:
                print(f"Época {ep:3d} | loss={tot/max(nb,1):.4f}  "
                      f"MAE={mae:6.1f}s  RMSE={rmse:6.1f}s  MAPE={mape:5.2f}%")
            if bad >= patience:
                if verbose:
                    print(f"Early stopping en época {ep}")
                break

        if best_w:
            self.model.load_state_dict(best_w)
        return self.history

    @torch.no_grad()
    def evaluate(self, cd: CityData, idx):
        """MAE y RMSE en segundos, MAPE en porcentaje (las tres métricas de los papers)."""
        if not idx:
            return float('nan'), float('nan'), float('nan')
        self.model.eval()
        ii = torch.tensor(idx, dtype=torch.long)
        h = self.model.encode_cells(*self._cells[cd.city_id])
        pred = self.model(h, cd.o_cell[ii].to(self.device),
                          cd.d_cell[ii].to(self.device),
                          cd.feats[ii].to(self.device)).cpu()
        secs = torch.exp(pred * cd.time_log_std + cd.time_log_mean)
        true = cd.raw_times[ii]
        err = secs - true
        return (float(err.abs().mean()),
                float(torch.sqrt((err ** 2).mean())),
                float((err.abs() / true.clamp(min=1)).mean() * 100))

    @torch.no_grad()
    def evaluate_by_hour(self, cd: CityData, idx):
        """
        Métricas por franja horaria, con el PEOR grupo. OOD4TTE reporta el peor
        caso además del promedio, y con razón: una mejora del promedio puede
        esconder un empeoramiento severo en la hora punta, que es justo cuando
        una estimación de tiempo de viaje importa.
        """
        buckets = {}
        hs = cd.feats[:, F_ROUTE]           # hour_sin
        hc = cd.feats[:, F_ROUTE + 1]       # hour_cos
        for i in idx:
            ang = math.atan2(float(hs[i]), float(hc[i])) % (2 * math.pi)
            h = int(ang * 24 / (2 * math.pi)) % 24
            buckets.setdefault(h // 6, []).append(i)
        out = {}
        for b, ids in sorted(buckets.items()):
            out[f"{b*6:02d}-{b*6+5:02d}h"] = self.evaluate(cd, ids)
        if out:
            out['peor_grupo'] = max(out.values(), key=lambda m: m[0])
        return out

    def save(self, path, quiet=False):
        torch.save({'model_state': self.model.state_dict(),
                    'config': self.model.config,
                    'norm': {'time_log_mean': self.target.time_log_mean,
                             'time_log_std': self.target.time_log_std},
                    'history': self.history}, path)
        if not quiet:
            print(f"Modelo guardado → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(target_csv, source_csvs=(), target_name=None, source_names=None,
                 target_contexts=None, source_contexts=None,
                 cell_m=500.0, split='chronological', split_kwargs=None,
                 epochs=60, device='auto', dim=64, hidden=128,
                 invariant=True, use_tsb=True, use_npm=True,
                 alpha=0.5, beta=0.5, target_fraction=None,
                 save_path=None, verbose=True, **loader_kwargs):
    """
    Entrena en una sola etapa sobre objetivo + fuentes y evalúa en la objetivo.

    target_fraction: si se indica, recorta el entrenamiento de la ciudad objetivo
                     a esa fracción. Es el régimen few-shot de TransferTraj, que
                     usa 5.000 trayectorias de la ciudad destino.
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    source_names = source_names or [None] * len(source_csvs)
    source_contexts = source_contexts or [None] * len(source_csvs)

    tgt_city = tl.CityTrips(
        tl.load_trips(target_csv, city_name=target_name, verbose=verbose,
                      **loader_kwargs),
        target_name or 'target', city_id=0, cell_m=cell_m,
        urban_ctx=target_contexts, verbose=verbose)
    src_cities = []
    for k, (p, n, c) in enumerate(zip(source_csvs, source_names, source_contexts)):
        src_cities.append(tl.CityTrips(
            tl.load_trips(p, city_name=n, verbose=verbose, **loader_kwargs),
            n or f'source{k}', city_id=k + 1, cell_m=cell_m,
            urban_ctx=c, verbose=verbose))

    dm, ds, tm, ts = global_norms([tgt_city] + src_cities)
    target = build_city_data(tgt_city, 0, dm, ds, tm, ts, split, split_kwargs)
    sources = [build_city_data(c, i + 1, dm, ds, tm, ts, 'chronological')
               for i, c in enumerate(src_cities)]

    if target_fraction is not None:
        k = max(int(len(target.train_idx) * target_fraction), 8)
        target.train_idx = target.train_idx[:k]
        if verbose:
            print(f"Régimen few-shot: {k} viajes de entrenamiento en la objetivo.")

    bank = max(target.n_cells // 2, 8)
    model = TravelTimeModel(dim=dim, hidden=hidden, bank_size=bank,
                            invariant=invariant, use_npm=use_npm)
    tr = TSJTTrainer(model, target, sources, device=device,
                     alpha=alpha, beta=beta, use_tsb=use_tsb)
    tr.train(epochs=epochs, verbose=verbose, checkpoint_path=save_path)

    mae, rmse, mape = tr.evaluate(target, target.test_idx)
    print(f"\n── Test en {target.city} ──")
    print(f"  MAE  {mae:7.1f} s ({mae/60:.2f} min)")
    print(f"  RMSE {rmse:7.1f} s ({rmse/60:.2f} min)")
    print(f"  MAPE {mape:7.2f} %")
    return tr, target


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    if args and args[0] == '--demo':
        paths, ctxs = [], []
        for i, n in enumerate(['Porto', 'Chengdu', 'Xian']):
            p = tl.write_synthetic_csv(f'/tmp/_tte_{n}.csv', n, n_trips=1500,
                                       seed=i, center=(41 - i * 5, -8 + i * 40))
            paths.append(p)
            trips = tl.load_trips(p, city_name=n, verbose=False)
            city = tl.CityTrips(trips, n, verbose=False)
            ctxs.append(urban_context.synthetic_context(city.cells, seed=i, city=n))
        # objetivo = Porto con pocos datos; fuentes = las otras dos
        run_pipeline(paths[0], paths[1:], target_name='Porto',
                     source_names=['Chengdu', 'Xian'],
                     target_contexts=ctxs[0], source_contexts=ctxs[1:],
                     epochs=30, target_fraction=0.1, verbose=True)
    else:
        print("Uso: python tte_model.py --demo")
        print("     run_pipeline('porto.csv', ['chengdu.csv'], ...)")