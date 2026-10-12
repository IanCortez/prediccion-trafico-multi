#!/usr/bin/env python3
"""
gnn_baseline_trips.py
=====================

Adapta el GNN de tiempos de viaje basado en GTFS (gnn_travel_time.py) para que
consuma los datasets de trayectorias (Oporto / Roma / Pekín) y sirva como
BASELINE comparable frente al modelo nuevo (tte_model.py).

Idea
----
El modelo antiguo no depende de GTFS: depende de un `feed` con dos listas,
`nodes` y `edges`. Este módulo fabrica ese feed desde los viajes, de modo que
NO se modifica ni una línea del modelo previo. La traducción es:

    GTFS                          Trayectorias
    ────────────────────────────  ──────────────────────────────────────────
    parada                        celda de la rejilla (misma que usa el nuevo)
    tramo de servicio             arista espacial (k vecinos) y de flujo O-D
    trayecto con tiempo medido    UN VIAJE, con su travel_time como etiqueta
    route_type                    3 (vehículo por carretera) para todo taxi
    headway de la parada          tiempo medio entre salidas en la celda
    n_routes                      nº de destinos distintos desde la celda

Comparabilidad
--------------
Para que la comparación signifique algo, el baseline comparte con el modelo
nuevo TODO lo que no es el modelo:

1. Los mismos viajes: se cargan con trajectory_loader.load_trips, con los
   mismos filtros del Algoritmo 1.
2. La misma rejilla: tl.CityTrips con el mismo cell_m.
3. Los mismos splits: chronological / ood_distance / ood_hour, calculados por
   el propio CityTrips, no por el sorteo aleatorio interno del Trainer antiguo.
4. Las mismas métricas: MAE y RMSE en segundos y MAPE en %, más el desglose
   por franja horaria con el PEOR grupo.
5. La misma prohibición de fuga: las features de nodo (degree, headway,
   betweenness, n_routes, n_support) se calculan SOLO con train_idx. Sin esto
   el baseline vería el test y la comparación estaría amañada a su favor.

Lo que el baseline NO tiene, y es justamente lo que se quiere medir: contexto
urbano OSM por celda, ST-GNN, NPM, gate de invariancia y transferencia por
prompts. Sí conserva su propio mecanismo de transferencia (discriminador de
ciudad adversarial), para que la comparación en transferencia sea honesta.

Uso
---
    # baseline en una ciudad
    python gnn_baseline_trips.py run --data data/roma/trips.csv --city roma \\
           --split chronological --epochs 150

    # baseline en transferencia (fuentes -> objetivo), adversarial activo
    python gnn_baseline_trips.py run --data data/pekin/trips.csv --city pekin \\
           --sources data/porto/train.csv data/roma/trips.csv \\
           --source-names porto roma --split ood_distance

    # tabla comparativa contra los runs del modelo nuevo
    python gnn_baseline_trips.py compare --city roma --split chronological

Dependencias: torch, numpy, trajectory_loader.py, gnn_travel_time.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import types
from collections import defaultdict, deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import trajectory_loader as tl

ROUTE_TYPE_TAXI = 3          # el modelo antiguo mapea 3 -> 'Bus' (carretera)
RUNS_DIR = 'runs'


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DEL MODELO ANTIGUO
# ─────────────────────────────────────────────────────────────────────────────

def load_legacy_gnn(path: str = 'gnn_travel_time.py') -> types.ModuleType:
    """
    Importa el módulo del GNN previo desde su ruta.

    El módulo hace `import gtfs_loader` en la cabecera. Aquí no se lee ningún
    GTFS, así que si el loader no está disponible se inyecta un stub: evita
    arrastrar una dependencia que este baseline no usa.
    """
    if 'gtfs_loader' not in sys.modules:
        try:
            import gtfs_loader  # noqa: F401
        except ImportError:
            stub = types.ModuleType('gtfs_loader')

            def _no(*_a, **_k):
                raise RuntimeError(
                    "Este baseline no lee GTFS; usa trips_to_feed().")
            stub.load_gtfs_feed = _no
            sys.modules['gtfs_loader'] = stub

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No encuentro el modelo previo en {path!r}. "
            f"Indícalo con --gnn-module (el fichero puede llamarse 'gnn-travel').")

    spec = importlib.util.spec_from_file_location('legacy_gnn', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['legacy_gnn'] = mod
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS DE GRAFO SOBRE LAS CELDAS (solo con train)
# ─────────────────────────────────────────────────────────────────────────────

def _knn_spatial(cells: List[dict], k: int, max_radius_m: float
                 ) -> List[Tuple[int, int, float]]:
    """k celdas más próximas de cada celda dentro de un radio. O(N²) honesto:
    para las pocas miles de celdas de una ciudad es suficiente."""
    n = len(cells)
    lat = np.array([c['lat'] for c in cells])
    lon = np.array([c['lon'] for c in cells])
    out = []
    for i in range(n):
        dlat = (lat - lat[i]) * 110_540.0
        dlon = (lon - lon[i]) * 111_320.0 * math.cos(math.radians(lat[i]))
        d = np.sqrt(dlat ** 2 + dlon ** 2)
        d[i] = np.inf
        cand = np.argsort(d)[:k]
        for j in cand:
            if d[j] <= max_radius_m:
                out.append((i, int(j), float(d[j])))
    return out


def _brandes_sampled(adj: Dict[int, set], n: int, n_pivots: int,
                     seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Betweenness y closeness aproximadas por muestreo de pivotes (Brandes).

    Exacto costaría O(N·E) y aquí no aporta: estas features solo describen la
    posición relativa de la celda en la red, y el error de muestreo es mucho
    menor que el ruido de los propios datos GPS.
    """
    rng = random.Random(seed)
    nodes = [i for i in range(n) if adj.get(i)]
    if not nodes:
        return np.zeros(n), np.zeros(n)
    pivots = nodes if len(nodes) <= n_pivots else rng.sample(nodes, n_pivots)

    btw = np.zeros(n)
    close_sum = np.zeros(n)
    close_cnt = np.zeros(n)

    for s in pivots:
        stack, pred = [], defaultdict(list)
        sigma = np.zeros(n); sigma[s] = 1.0
        dist = np.full(n, -1.0); dist[s] = 0.0
        q = deque([s])
        while q:
            v = q.popleft()
            stack.append(v)
            for w in adj.get(v, ()):
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        reach = dist > 0
        close_sum[s] += dist[reach].sum()
        close_cnt[s] += reach.sum()

        delta = np.zeros(n)
        for w in reversed(stack):
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                btw[w] += delta[w]

    scale = len(pivots) * max(n - 1, 1)
    btw /= max(scale, 1.0)
    btw = btw / (btw.max() + 1e-9)
    closeness = np.where(close_sum > 0, close_cnt / np.maximum(close_sum, 1e-9), 0.0)
    closeness = closeness / (closeness.max() + 1e-9)
    return btw, closeness


def _clustering(adj: Dict[int, set], n: int) -> np.ndarray:
    out = np.zeros(n)
    for i in range(n):
        nb = adj.get(i, set())
        k = len(nb)
        if k < 2:
            continue
        links = sum(1 for u in nb for v in nb if u < v and v in adj.get(u, ()))
        out[i] = 2.0 * links / (k * (k - 1))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# VIAJES -> FEED DEL MODELO ANTIGUO
# ─────────────────────────────────────────────────────────────────────────────

def trips_to_feed(city: tl.CityTrips, train_idx: Sequence[int],
                  k_spatial: int = 8, max_radius_m: float = 2000.0,
                  min_flow: int = 3, n_pivots: int = 128,
                  verbose: bool = True) -> dict:
    """
    Traduce una ciudad de viajes al `feed` que espera _build_graph_data().

    Orden de las aristas, IMPORTANTE: primero los viajes (etiquetados, uno por
    viaje y en el orden de city.trips) y después las aristas estructurales sin
    etiqueta. Así el índice i de arista etiquetada coincide con el viaje i, y
    los splits del CityTrips se pueden aplicar tal cual sobre las aristas.
    """
    cells = city.cells
    n = len(cells)
    train = set(train_idx)

    # ── estadísticos de celda, SOLO con entrenamiento ────────────────────────
    out_dst: Dict[int, set] = defaultdict(set)     # destinos distintos
    salidas: Dict[int, List[int]] = defaultdict(list)
    flow: Dict[Tuple[int, int], int] = defaultdict(int)
    adj: Dict[int, set] = defaultdict(set)

    for i in train_idx:
        t = city.trips[i]
        o, d = t['o_cell'], t['d_cell']
        salidas[o].append(t['timestamp'])
        if o != d:
            out_dst[o].add(d)
            flow[(o, d)] += 1

    # aristas estructurales: vecindad espacial + flujo O-D observado
    espaciales = _knn_spatial(cells, k_spatial, max_radius_m)
    for i, j, _ in espaciales:
        adj[i].add(j); adj[j].add(i)
    funcionales = [(o, d, c) for (o, d), c in flow.items() if c >= min_flow]
    for o, d, _ in funcionales:
        adj[o].add(d); adj[d].add(o)

    btw, clo = _brandes_sampled(adj, n, n_pivots)
    clu = _clustering(adj, n)
    degree = np.array([len(adj.get(i, ())) for i in range(n)], dtype=np.float64)
    p90 = np.percentile(degree, 90) if n else 0.0

    # headway: tiempo medio entre salidas consecutivas observadas en la celda
    headway = np.full(n, 3600.0)
    for c, ts in salidas.items():
        if len(ts) >= 2:
            ts = sorted(ts)
            headway[c] = max((ts[-1] - ts[0]) / (len(ts) - 1), 1.0)

    lat_c, lon_c, R = city.lat_c, city.lon_c, max(city.radius, 1.0)

    nodes = []
    for i, c in enumerate(cells):
        nodes.append({
            'id': f'c{i}',
            'name': f'celda {i}',
            'lat': c['lat'], 'lon': c['lon'],
            'modes': [ROUTE_TYPE_TAXI],
            'dist_center': min(tl._haversine(c['lat'], c['lon'],
                                             lat_c, lon_c) / R, 2.0),
            'degree': float(degree[i]),
            'betweenness': float(btw[i]),
            'closeness': float(clo[i]),
            'clustering': float(clu[i]),
            'n_routes': float(len(out_dst.get(i, ()))),
            'headway': float(headway[i]),
            'is_terminal': 1.0 if degree[i] <= 1 else 0.0,
            'is_interchange': 1.0 if degree[i] >= p90 and p90 > 0 else 0.0,
        })

    # ── aristas ──────────────────────────────────────────────────────────────
    edges = []

    # 1) un viaje = una arista etiquetada. edge_type 'journey' para que NO
    #    entre en la matriz de adyacencia: conectar O-D lejanos rompería la
    #    agregación local del message passing (y filtraría el objetivo).
    for t in city.trips:
        o, d = t['o_cell'], t['d_cell']
        edges.append({
            'source': f'c{o}', 'target': f'c{d}',
            'edge_type': 'journey',
            'route_type': ROUTE_TYPE_TAXI,
            'route_dist': float(t['od_dist']),
            'avg_dur': float(t['travel_time']),
            'dep_hour': float((t['timestamp'] // 3600) % 24),
            'n_support': int(flow.get((o, d), 0)),     # solo train
            'n_hops': 1, 'n_transfers': 0,
        })

    # 2) topología espacial (sin etiqueta): define la vecindad del encoder
    for i, j, dist in espaciales:
        edges.append({
            'source': f'c{i}', 'target': f'c{j}',
            'edge_type': 'service', 'route_type': ROUTE_TYPE_TAXI,
            'route_dist': dist, 'avg_dur': None, 'dep_hour': 12.0,
            'n_support': 1, 'n_hops': 1, 'n_transfers': 0,
        })

    # 3) corredores funcionales observados en train (sin etiqueta)
    for o, d, c in funcionales:
        la1, lo1 = cells[o]['lat'], cells[o]['lon']
        la2, lo2 = cells[d]['lat'], cells[d]['lon']
        edges.append({
            'source': f'c{o}', 'target': f'c{d}',
            'edge_type': 'service', 'route_type': ROUTE_TYPE_TAXI,
            'route_dist': tl._haversine(la1, lo1, la2, lo2),
            'avg_dur': None, 'dep_hour': 12.0,
            'n_support': c, 'n_hops': 1, 'n_transfers': 0,
        })

    if verbose:
        print(f"[feed:{city.city}] {n} celdas | {len(city.trips)} viajes "
              f"(aristas etiquetadas) | {len(espaciales)} espaciales | "
              f"{len(funcionales)} funcionales")
    return {'nodes': nodes, 'edges': edges, 'city': city.city}


# ─────────────────────────────────────────────────────────────────────────────
# SPLITS COMPARTIDOS CON EL MODELO NUEVO
# ─────────────────────────────────────────────────────────────────────────────

def make_splits(city: tl.CityTrips, split: str, **kw):
    """Replica exactamente build_city_data() del pipeline nuevo."""
    if split == 'chronological':
        return city.chronological_split(**kw)
    if split == 'ood_hour':
        tr, te = city.ood_split_by_hour(**kw)
    elif split == 'ood_distance':
        tr, te = city.ood_split_by_distance(**kw)
    else:
        raise ValueError(f"split desconocido: {split!r}")
    cut = int(len(tr) * 0.9)
    return tr[:cut], tr[cut:], te


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL GraphData
# ─────────────────────────────────────────────────────────────────────────────

def build_baseline_data(legacy, cities: List[tl.CityTrips],
                        splits: List[Tuple[list, list, list]], **feed_kw):
    """
    Devuelve (GraphData, offsets). La ciudad objetivo debe ir la PRIMERA: sus
    aristas etiquetadas ocupan las posiciones [0, n_viajes_objetivo), de modo
    que los índices de split del objetivo son válidos sin desplazamiento.
    """
    feeds = [trips_to_feed(c, s[0], **feed_kw) for c, s in zip(cities, splits)]
    dm, ds, tlm, tls = legacy._global_norms(feeds)
    graphs = [legacy._build_graph_data(f, city_id=i, dist_mean=dm, dist_std=ds,
                                       time_log_mean=tlm, time_log_std=tls)
              for i, f in enumerate(feeds)]

    # comprobación dura: sin esta correspondencia los splits no significan nada
    for g, c in zip(graphs, cities):
        if len(g.edge_times) != len(c.trips):
            raise RuntimeError(
                f"[{c.city}] {len(g.edge_times)} aristas etiquetadas != "
                f"{len(c.trips)} viajes; la correspondencia viaje↔arista se rompió.")

    data = graphs[0] if len(graphs) == 1 else legacy._merge(graphs, len(graphs))
    offsets, acc = [], 0
    for c in cities:
        offsets.append(acc)
        acc += len(c.trips)
    return data, offsets


# ─────────────────────────────────────────────────────────────────────────────
# TRAINER CON SPLITS FIJOS Y MÉTRICAS DEL PAPER
# ─────────────────────────────────────────────────────────────────────────────

class BaselineRunner:
    """
    Envuelve el Trainer antiguo para imponerle los splits del modelo nuevo y
    exponer las mismas métricas (MAE, RMSE en s, MAPE en %).

    El Trainer original parte train/val con un shuffle aleatorio. Con viajes de
    taxi eso es fuga: dos viajes del mismo taxi en la misma hora acaban uno en
    train y otro en val, y la validación deja de medir generalización.
    """

    def __init__(self, legacy, data, device='cpu', lr=1e-3,
                 adversarial=True, adv_weight=0.1, **model_kw):
        self.legacy = legacy
        self.data = data
        self.model = legacy.TravelTimeGNN(**model_kw)
        self.trainer = legacy.Trainer(self.model, data, device=device, lr=lr,
                                      adversarial=adversarial,
                                      adv_weight=adv_weight)

    def set_splits(self, train_idx, val_idx):
        d, T = self.data, self.trainer
        tr = torch.tensor(list(train_idx), dtype=torch.long)
        va = torch.tensor(list(val_idx), dtype=torch.long)
        T.train_ds = self.legacy.EdgeDataset(d.edge_src[tr], d.edge_dst[tr],
                                             d.edge_feats[tr], d.edge_times[tr])
        T.val_ds = self.legacy.EdgeDataset(d.edge_src[va], d.edge_dst[va],
                                           d.edge_feats[va], d.edge_times[va])
        return self

    def train(self, epochs=150, batch_size=256, patience=20, verbose=True):
        self.trainer.train(epochs=epochs, batch_size=batch_size,
                           patience=patience, verbose=verbose)
        return self

    @torch.no_grad()
    def _pred_seconds(self, idx):
        d, T = self.data, self.trainer
        ii = torch.tensor(list(idx), dtype=torch.long)
        h = T._encode()
        pn = T._predict_chunked(h, d.edge_src[ii], d.edge_dst[ii], d.edge_feats[ii])
        return T._invert(pn), T._invert(d.edge_times[ii])

    def evaluate(self, idx):
        """MAE y RMSE en segundos, MAPE en %. Idéntico a TSJTTrainer.evaluate."""
        if len(idx) == 0:
            return float('nan'), float('nan'), float('nan')
        pred, true = self._pred_seconds(idx)
        err = pred - true
        return (float(err.abs().mean()),
                float(torch.sqrt((err ** 2).mean())),
                float((err.abs() / true.clamp(min=1)).mean() * 100))

    def evaluate_by_hour(self, city: tl.CityTrips, idx, offset=0):
        """Desglose por franja de 6 h y peor grupo, como el modelo nuevo."""
        buckets = defaultdict(list)
        for i in idx:
            h = int((city.trips[i - offset]['timestamp'] // 3600) % 24)
            buckets[h // 6].append(i)
        out = {f"{b*6:02d}-{b*6+5:02d}h": self.evaluate(ids)
               for b, ids in sorted(buckets.items())}
        if out:
            out['peor_grupo'] = max(out.values(), key=lambda m: m[0])
        return out


# ─────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    legacy = load_legacy_gnn(args.gnn_module)

    # objetivo primero; las fuentes solo aportan datos al adversarial
    rutas = [args.data] + list(args.sources or [])
    nombres = [args.city] + list(args.source_names or
                                 [None] * len(args.sources or []))
    cities, splits = [], []
    for k, (p, nm) in enumerate(zip(rutas, nombres)):
        trips = tl.load_trips(p, city_name=nm, max_trips=args.max_trips)
        c = tl.CityTrips(trips, nm or f'city{k}', city_id=k, cell_m=args.cell_m)
        cities.append(c)
        # las ciudades fuente siempre van en split cronológico, como en el nuevo
        splits.append(make_splits(c, args.split if k == 0 else 'chronological',
                                  **(_split_kwargs(args) if k == 0 else {})))

    data, offsets = build_baseline_data(
        legacy, cities, splits, k_spatial=args.graph_k,
        max_radius_m=args.max_radius, min_flow=args.min_flow)

    # índices globales: objetivo en [0, n0); fuentes desplazadas
    tr0, va0, te0 = splits[0]
    train_idx = list(tr0)
    for k in range(1, len(cities)):
        train_idx += [i + offsets[k] for i in splits[k][0]]

    device = (args.device if args.device != 'auto'
              else ('cuda' if torch.cuda.is_available() else 'cpu'))
    runner = BaselineRunner(legacy, data, device=device, lr=args.lr,
                            adversarial=not args.no_adversarial,
                            adv_weight=args.adv_weight,
                            node_dim=args.node_dim, hidden=args.hidden,
                            n_layers=args.layers, dropout=args.dropout)
    runner.set_splits(train_idx, va0)

    print(f"train/val/test = {len(train_idx)}/{len(va0)}/{len(te0)}"
          f"  | celdas {len(data.node_ids)}"
          f"  | aristas etiquetadas {len(data.edge_times)}"
          f"  | ciudades {data.n_cities}")

    runner.train(epochs=args.epochs, batch_size=args.batch,
                 patience=args.patience)

    mae, rmse, mape = runner.evaluate(te0)
    print(f"\n── TEST {cities[0].city} · split={args.split} · BASELINE GNN-GTFS ──")
    print(f"  MAE {mae:8.1f} s | RMSE {rmse:8.1f} s | MAPE {mape:6.2f} %")
    print("\n── por franja horaria ──")
    for k, v in runner.evaluate_by_hour(cities[0], te0).items():
        print(f"  {k:<12} MAE {v[0]:7.1f} s | MAPE {v[2]:6.2f} %")

    os.makedirs(RUNS_DIR, exist_ok=True)
    tag = f"baseline_{args.city}_{args.split}_s{args.seed}"
    runner.trainer.save(os.path.join(RUNS_DIR, f'{tag}.pt'))
    with open(os.path.join(RUNS_DIR, f'{tag}.json'), 'w', encoding='utf-8') as f:
        json.dump({'model': 'gnn_gtfs_baseline', 'mae': mae, 'rmse': rmse,
                   'mape': mape, 'args': vars(args)}, f, indent=2, default=str)
    return runner


def _split_kwargs(args) -> dict:
    if args.split == 'ood_hour':
        return {'held_out_hours': args.hours}
    if args.split == 'ood_distance':
        return {'quantile': args.quantile}
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# COMPARACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def compare(args):
    """Tabla baseline vs modelo nuevo leyendo los JSON de runs/."""
    filas = []
    for split in (args.splits or ['chronological', 'ood_distance', 'ood_hour']):
        base = os.path.join(RUNS_DIR, f'baseline_{args.city}_{split}_s{args.seed}.json')
        nuevo = os.path.join(RUNS_DIR, f'{args.city}_{split}_s{args.seed}.json')
        if not (os.path.exists(base) and os.path.exists(nuevo)):
            continue
        b = json.load(open(base, encoding='utf-8'))
        n = json.load(open(nuevo, encoding='utf-8'))
        filas.append((split, b, n))

    if not filas:
        print(f"No hay pares de runs para {args.city} en {RUNS_DIR}/.")
        return

    print(f"\n{'='*76}")
    print(f"  {args.city.upper()} — BASELINE (GNN GTFS adaptado) vs MODELO NUEVO")
    print(f"{'='*76}")
    print(f"  {'split':<16}{'MAE base':>11}{'MAE nuevo':>11}{'Δ':>9}"
          f"{'MAPE base':>11}{'MAPE nuevo':>12}")
    print(f"  {'-'*72}")
    for split, b, n in filas:
        dmae = 100.0 * (b['mae'] - n['mae']) / b['mae'] if b['mae'] else 0.0
        print(f"  {split:<16}{b['mae']:>10.1f}s{n['mae']:>10.1f}s"
              f"{dmae:>8.1f}%{b['mape']:>10.2f}%{n['mape']:>11.2f}%")
    print(f"  {'-'*72}")
    print("  Δ = reducción de MAE del modelo nuevo sobre el baseline "
          "(positivo = mejora).\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(
        description='Baseline: GNN de GTFS adaptado a datasets de trayectorias.')
    sub = p.add_subparsers(dest='cmd', required=True)

    r = sub.add_parser('run', help='entrena y evalúa el baseline')
    r.add_argument('--data', required=True, help='CSV de viajes de la ciudad objetivo')
    r.add_argument('--city', required=True)
    r.add_argument('--sources', nargs='*', default=[],
                   help='CSV de ciudades fuente (activan el adversarial)')
    r.add_argument('--source-names', nargs='*', default=[])
    r.add_argument('--gnn-module', default='gnn_travel_time.py',
                   help="ruta del modelo previo (p. ej. 'gnn-travel')")
    r.add_argument('--split', default='chronological',
                   choices=['chronological', 'ood_distance', 'ood_hour'])
    r.add_argument('--hours', nargs='*', type=int, default=[7, 8, 9])
    r.add_argument('--quantile', type=float, default=0.7)
    r.add_argument('--cell-m', type=float, default=500.0)
    r.add_argument('--graph-k', type=int, default=8)
    r.add_argument('--max-radius', type=float, default=2000.0)
    r.add_argument('--min-flow', type=int, default=3)
    r.add_argument('--max-trips', type=int, default=None)
    r.add_argument('--epochs', type=int, default=150)
    r.add_argument('--batch', type=int, default=256)
    r.add_argument('--patience', type=int, default=20)
    r.add_argument('--lr', type=float, default=1e-3)
    r.add_argument('--node-dim', type=int, default=32)
    r.add_argument('--hidden', type=int, default=64)
    r.add_argument('--layers', type=int, default=2)
    r.add_argument('--dropout', type=float, default=0.1)
    r.add_argument('--adv-weight', type=float, default=0.1)
    r.add_argument('--no-adversarial', action='store_true')
    r.add_argument('--device', default='auto')
    r.add_argument('--seed', type=int, default=0)
    r.set_defaults(func=run)

    c = sub.add_parser('compare', help='tabla baseline vs modelo nuevo')
    c.add_argument('--city', required=True)
    c.add_argument('--splits', nargs='*')
    c.add_argument('--seed', type=int, default=0)
    c.set_defaults(func=compare)

    a = p.parse_args(argv)
    a.func(a)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())