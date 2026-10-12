"""
gnn_travel_time.py  (versión adaptada para transferencia entre ciudades + GTFS)
================================================================================
Predice tiempos de viaje entre paradas y, sobre todo, TRANSFIERE ese conocimiento
de ciudades con datos GTFS hacia ciudades con pocos datos o ninguno.
 
Cambios frente a la versión original
------------------------------------
1. CONSUMO DE GTFS CRUDO
   extract_from_gtfs() / extract_from_multiple_gtfs() leen feeds GTFS mediante
   gtfs_loader.py — ya no se necesita un graph_data.json intermedio.
 
2. FEATURES ENRIQUECIDAS Y TRANSFERIBLES
   El encoder de nodo deja de depender solo de (lat, lon). Ahora usa
   conectividad topológica (degree, betweenness, closeness, clustering),
   intensidad de servicio (n_routes, headway) y banderas estructurales.
   Las aristas incorporan codificación temporal (sin/cos de la hora).
   Ver NODE_FEATURE_NAMES y EDGE_FEATURE_NAMES.
 
3. ENTRENAMIENTO ADVERSARIAL DOMAIN-INVARIANT
   Un CityDiscriminator intenta adivinar la ciudad de cada embedding; mediante
   un Gradient Reversal Layer el encoder se entrena para ENGAÑARLO. Así los
   embeddings dejan de codificar la identidad de la ciudad y solo conservan
   información funcional reutilizable (mecanismo principal de transferencia).
 
4. OBJETIVO EN ESPACIO LOGARÍTMICO
   Los tiempos de viaje son positivos y sesgados a la derecha: se entrena sobre
   log(tiempo), lo que evita sobreestimar tramos cortos y subestimar largos.
 
Notas de alcance: la fusión semántica multi-vista y el clustering jerárquico
multi-granularidad de SFMGTL quedan como puntos de extensión documentados; aquí
se implementan los componentes de mayor impacto para el caso pedido.
 
Dependencias: torch, numpy, gtfs_loader (incluido).
"""
 
import os
import math
import json
import random
from dataclasses import dataclass, field
from typing import Optional, List
 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
 
import gtfs_loader
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
 
MODE_TO_IDX = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
               5: 5, 6: 5, 7: 5, 11: 3, 12: 1}
N_MODES = 6
 
ROUTE_TYPE_NAMES = {
    0: 'Tranvía', 1: 'Metro', 2: 'Ferrocarril', 3: 'Bus', 4: 'Ferry',
    5: 'Teleférico', 6: 'Góndola', 7: 'Funicular', 11: 'Trolebús', 12: 'Monorraíl',
}
 
# Orden de las columnas del vector de features de nodo (antes del one-hot de modo)
NODE_FEATURE_NAMES = [
    'lat_norm', 'lon_norm', 'dist_center', 'degree', 'betweenness',
    'closeness', 'clustering', 'n_modes', 'n_routes', 'headway',
    'is_terminal', 'is_interchange',
]
F_NODE = len(NODE_FEATURE_NAMES)            # 12
 
# Orden de las columnas del vector de features de arista (antes del one-hot de modo)
#   n_hops      = nº de paradas intermedias del tramo/trayecto (1 = un salto)
#   n_transfers = nº de transbordos del trayecto (0 salvo recorridos multi-etapa)
EDGE_FEATURE_NAMES = ['dist_norm', 'is_transfer', 'hour_sin', 'hour_cos',
                      'support', 'n_hops', 'n_transfers']
F_EDGE = len(EDGE_FEATURE_NAMES) + N_MODES  # 7 + 6 = 13
 
 
# ─────────────────────────────────────────────────────────────────────────────
# RESULTADO DE PREDICCIÓN
# ─────────────────────────────────────────────────────────────────────────────
 
@dataclass
class PredictionResult:
    src_id: str
    src_name: str
    src_lat: float
    src_lon: float
    src_modes: list
    dst_id: str
    dst_name: str
    dst_lat: float
    dst_lon: float
    dst_modes: list
    time_seconds: float
    time_minutes: float
    route_type: int
    route_name: str
    distance_m: float
 
    def __str__(self) -> str:
        sm = [ROUTE_TYPE_NAMES.get(m, str(m)) for m in self.src_modes]
        dm = [ROUTE_TYPE_NAMES.get(m, str(m)) for m in self.dst_modes]
        return (
            f"\n{'─'*58}\n"
            f"  ORIGEN : {self.src_name}\n"
            f"           ID={self.src_id}  ({self.src_lat:.5f}, {self.src_lon:.5f})\n"
            f"           Modos: {', '.join(sm)}\n"
            f"  DESTINO: {self.dst_name}\n"
            f"           ID={self.dst_id}  ({self.dst_lat:.5f}, {self.dst_lon:.5f})\n"
            f"           Modos: {', '.join(dm)}\n"
            f"  ─────────────────────────────────────────────────────\n"
            f"  Transporte: {self.route_name}\n"
            f"  Distancia:  {self.distance_m:,.0f} m\n"
            f"  Tiempo:     {self.time_seconds:.0f} s  ({self.time_minutes:.1f} min)\n"
            f"{'─'*58}"
        )
 
    def to_dict(self) -> dict:
        return {
            'src': {'id': self.src_id, 'name': self.src_name,
                    'lat': self.src_lat, 'lon': self.src_lon,
                    'modes': [ROUTE_TYPE_NAMES.get(m, str(m)) for m in self.src_modes]},
            'dst': {'id': self.dst_id, 'name': self.dst_name,
                    'lat': self.dst_lat, 'lon': self.dst_lon,
                    'modes': [ROUTE_TYPE_NAMES.get(m, str(m)) for m in self.dst_modes]},
            'transport': self.route_name,
            'distance_m': round(self.distance_m, 1),
            'time_s': round(self.time_seconds, 1),
            'time_min': round(self.time_minutes, 2),
        }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ESTRUCTURA TENSORIAL DEL GRAFO
# ─────────────────────────────────────────────────────────────────────────────
 
@dataclass
class GraphData:
    """Todo lo que el modelo necesita, ya normalizado y en tensores."""
    node_ids:   list
    node_feats: torch.Tensor          # [N, F_NODE]   features enriquecidas
    node_modes: torch.Tensor          # [N]           modo primario (índice)
    city_ids:   torch.Tensor          # [N]           índice de ciudad (discriminador)
 
    edge_src:   torch.Tensor          # [E_lab]
    edge_dst:   torch.Tensor          # [E_lab]
    edge_feats: torch.Tensor          # [E_lab, F_EDGE]
    edge_times: torch.Tensor          # [E_lab]   log-tiempo normalizado (objetivo)
 
    pred_src:   torch.Tensor          # [E_unlab]
    pred_dst:   torch.Tensor          # [E_unlab]
    pred_feats: torch.Tensor          # [E_unlab, F_EDGE]
 
    adj:        torch.Tensor          # [2, E_total]  para message passing
 
    node_meta:  dict = field(default_factory=dict)
    labeled_route_types:   list = field(default_factory=list)
    unlabeled_route_types: list = field(default_factory=list)
 
    n_cities:   int = 1
    # normalización (en espacio log para el tiempo)
    time_log_mean: float = 0.0
    time_log_std:  float = 1.0
    dist_mean:     float = 0.0
    dist_std:      float = 1.0
 
 
# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────
 
def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.asin(math.sqrt(a))
 
 
def primary_mode(modes: list) -> int:
    """Índice interno del modo más rápido presente en la parada."""
    priority = [2, 1, 12, 0, 11, 3, 4, 5, 6, 7]
    for m in priority:
        if m in modes:
            return MODE_TO_IDX.get(m, 3)
    return 3
 
 
def _estimate_n_hops(distance_m, avg_spacing_m=600.0):
    """
    Estima el nº de paradas intermedias a partir de la distancia, para
    consultas donde no se conoce el recorrido real (predicción por distancia).
    """
    return max(1, round(distance_m / max(avg_spacing_m, 1.0)))
 
 
def _edge_feature_vector(dist_m, route_type, is_transfer, dep_hour,
                         n_support, n_hops, n_transfers, dist_mean, dist_std):
    """Construye el vector de features de una arista (F_EDGE dimensiones)."""
    dist_norm = (dist_m - dist_mean) / dist_std
    sin_h = math.sin(2 * math.pi * dep_hour / 24.0)
    cos_h = math.cos(2 * math.pi * dep_hour / 24.0)
    support = min(math.log1p(n_support) / 4.0, 1.5)
    hops = min(math.log1p(max(n_hops, 0.0)) / 4.0, 2.0)
    one_hot = [0.0] * N_MODES
    one_hot[MODE_TO_IDX.get(route_type, 3)] = 1.0
    return [dist_norm, 1.0 if is_transfer else 0.0, sin_h, cos_h,
            support, hops, float(n_transfers)] + one_hot
 
 
# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZACIÓN GLOBAL (compartida entre ciudades)
# ─────────────────────────────────────────────────────────────────────────────
 
def _global_norms(feeds: List[dict]):
    """
    Calcula la normalización de distancia y de log-tiempo sobre TODAS las
    ciudades juntas, para que las features sean comparables en la transferencia.
    Usa route_dist (distancia de ruta real) cuando está disponible.
    """
    dists, log_times = [], []
    for feed in feeds:
        coord = {n['id']: (n['lat'], n['lon']) for n in feed['nodes']}
        for e in feed['edges']:
            if e['source'] not in coord or e['target'] not in coord:
                continue
            d = e.get('route_dist')
            if d is None:
                la1, lo1 = coord[e['source']]
                la2, lo2 = coord[e['target']]
                d = haversine(la1, lo1, la2, lo2)
            dists.append(d)
            if e.get('avg_dur') and e['avg_dur'] > 0:
                log_times.append(math.log(e['avg_dur']))
 
    dists = np.array(dists, dtype=np.float64) if dists else np.array([1.0])
    log_times = np.array(log_times, dtype=np.float64) if log_times else np.array([0.0])
    return (float(dists.mean()), float(dists.std() + 1e-8),
            float(log_times.mean()), float(log_times.std() + 1e-8))
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE GraphData DESDE UN FEED
# ─────────────────────────────────────────────────────────────────────────────
 
def _build_graph_data(feed: dict, city_id: int,
                      dist_mean: float, dist_std: float,
                      time_log_mean: float, time_log_std: float) -> GraphData:
    """Núcleo: convierte un feed (nodes/edges enriquecidos) en GraphData."""
    nodes_raw = feed['nodes']
    edges_raw = feed['edges']
 
    node_ids = [n['id'] for n in nodes_raw]
    node_idx = {nid: i for i, nid in enumerate(node_ids)}
    N = len(node_ids)
 
    # ── features de nodo (z-score por ciudad para columnas no acotadas) ───────
    def col(name, default=0.0):
        return np.array([n.get(name, default) for n in nodes_raw], dtype=np.float32)
 
    def z(a):
        return (a - a.mean()) / (a.std() + 1e-8)
 
    lat = col('lat'); lon = col('lon')
    degree = col('degree'); n_routes = col('n_routes')
    headway_log = np.log1p(col('headway', 3600.0))
    n_modes = np.array([len(set(n.get('modes', [3]))) for n in nodes_raw],
                       dtype=np.float32)
 
    node_feats = np.stack([
        z(lat),                       # lat_norm     (específica de ciudad)
        z(lon),                       # lon_norm     (específica de ciudad)
        col('dist_center'),           # dist_center  ∈[0,1]  transferible
        z(degree),                    # degree       transferible
        col('betweenness'),           # betweenness  transferible
        col('closeness'),             # closeness    transferible
        col('clustering'),            # clustering   transferible
        n_modes / N_MODES,            # n_modes      transferible
        z(n_routes),                  # n_routes     transferible
        z(headway_log),               # headway      transferible
        col('is_terminal'),           # bandera estructural
        col('is_interchange'),        # bandera estructural
    ], axis=1).astype(np.float32)
 
    modes = np.array([primary_mode(n.get('modes', [3])) for n in nodes_raw],
                     dtype=np.int64)
 
    node_meta = {
        n['id']: {'name': n.get('name', n['id']), 'lat': n['lat'],
                  'lon': n['lon'], 'modes': n.get('modes', [3])}
        for n in nodes_raw
    }
    coord = {n['id']: (n['lat'], n['lon']) for n in nodes_raw}
 
    # ── separar aristas etiquetadas / sin etiquetar ───────────────────────────
    #   El message passing (adj) usa SOLO la topología física: tramos
    #   consecutivos y transbordos. Los trayectos O-D ('journey') son etiquetas
    #   de entrenamiento pero NO entran en adj (conectarían paradas lejanas y
    #   romperían la agregación local de vecinos).
    labeled, unlabeled, all_adj = [], [], set()
    for e in edges_raw:
        u, v = e['source'], e['target']
        if u not in node_idx or v not in node_idx:
            continue
        si, di = node_idx[u], node_idx[v]
        edge_type = e.get('edge_type', 'service')
        if edge_type in ('service', 'transfer'):
            all_adj.add((si, di))                 # solo topología física
 
        # distancia de ruta real si está disponible; si no, línea recta
        dist = e.get('route_dist')
        if dist is None:
            la1, lo1 = coord[u]; la2, lo2 = coord[v]
            dist = haversine(la1, lo1, la2, lo2)
        rt = e.get('route_type', 3)
        is_tr = edge_type == 'transfer'
        feats = _edge_feature_vector(
            dist, rt, is_tr, e.get('dep_hour', 12.0), e.get('n_support', 1),
            e.get('n_hops', 1), e.get('n_transfers', 0), dist_mean, dist_std)
        avg = e.get('avg_dur')
        if avg and avg > 0:
            t_norm = (math.log(avg) - time_log_mean) / time_log_std
            labeled.append((si, di, feats, t_norm, rt))
        else:
            unlabeled.append((si, di, feats, rt))
 
    def to_tensors(rows, has_time):
        if not rows:
            empty_t = torch.zeros(0, dtype=torch.long)
            return (empty_t, empty_t, torch.zeros(0, F_EDGE),
                    torch.zeros(0) if has_time else None, [])
        src = torch.tensor([r[0] for r in rows], dtype=torch.long)
        dst = torch.tensor([r[1] for r in rows], dtype=torch.long)
        fts = torch.tensor([r[2] for r in rows], dtype=torch.float32)
        if has_time:
            tms = torch.tensor([r[3] for r in rows], dtype=torch.float32)
            rts = [r[4] for r in rows]
            return src, dst, fts, tms, rts
        rts = [r[3] for r in rows]
        return src, dst, fts, None, rts
 
    l_src, l_dst, l_feats, l_times, l_rts = to_tensors(labeled, True)
    p_src, p_dst, p_feats, _, p_rts = to_tensors(unlabeled, False)
 
    adj = (torch.tensor(list(all_adj), dtype=torch.long).T
           if all_adj else torch.zeros(2, 0, dtype=torch.long))
 
    return GraphData(
        node_ids=node_ids,
        node_feats=torch.tensor(node_feats),
        node_modes=torch.tensor(modes),
        city_ids=torch.full((N,), city_id, dtype=torch.long),
        edge_src=l_src, edge_dst=l_dst, edge_feats=l_feats,
        edge_times=l_times if l_times is not None else torch.zeros(0),
        pred_src=p_src, pred_dst=p_dst, pred_feats=p_feats,
        adj=adj,
        node_meta=node_meta,
        labeled_route_types=l_rts, unlabeled_route_types=p_rts,
        n_cities=1,
        time_log_mean=time_log_mean, time_log_std=time_log_std,
        dist_mean=dist_mean, dist_std=dist_std,
    )
 
 
def _merge(graphs: List[GraphData], n_cities: int) -> GraphData:
    """Fusiona varias ciudades desplazando índices de nodo."""
    node_ids, node_feats, node_modes, city_ids = [], [], [], []
    node_meta = {}
    e_src, e_dst, e_feats, e_times, e_rts = [], [], [], [], []
    p_src, p_dst, p_feats, p_rts = [], [], [], []
    adj_parts = []
 
    offset = 0
    for g in graphs:
        n = len(g.node_ids)
        node_ids.extend(g.node_ids)
        node_feats.append(g.node_feats)
        node_modes.append(g.node_modes)
        city_ids.append(g.city_ids)
        node_meta.update(g.node_meta)
 
        if g.edge_src.numel():
            e_src.append(g.edge_src + offset)
            e_dst.append(g.edge_dst + offset)
            e_feats.append(g.edge_feats)
            e_times.append(g.edge_times)
            e_rts.extend(g.labeled_route_types)
        if g.pred_src.numel():
            p_src.append(g.pred_src + offset)
            p_dst.append(g.pred_dst + offset)
            p_feats.append(g.pred_feats)
            p_rts.extend(g.unlabeled_route_types)
        if g.adj.numel():
            adj_parts.append(g.adj + offset)
        offset += n
 
    def cat(parts, dim=0, fallback=None):
        return torch.cat(parts, dim=dim) if parts else fallback
 
    base = graphs[0]
    return GraphData(
        node_ids=node_ids,
        node_feats=cat(node_feats),
        node_modes=cat(node_modes),
        city_ids=cat(city_ids),
        edge_src=cat(e_src, fallback=torch.zeros(0, dtype=torch.long)),
        edge_dst=cat(e_dst, fallback=torch.zeros(0, dtype=torch.long)),
        edge_feats=cat(e_feats, fallback=torch.zeros(0, F_EDGE)),
        edge_times=cat(e_times, fallback=torch.zeros(0)),
        pred_src=cat(p_src, fallback=torch.zeros(0, dtype=torch.long)),
        pred_dst=cat(p_dst, fallback=torch.zeros(0, dtype=torch.long)),
        pred_feats=cat(p_feats, fallback=torch.zeros(0, F_EDGE)),
        adj=cat(adj_parts, dim=1, fallback=torch.zeros(2, 0, dtype=torch.long)),
        node_meta=node_meta,
        labeled_route_types=e_rts, unlabeled_route_types=p_rts,
        n_cities=n_cities,
        time_log_mean=base.time_log_mean, time_log_std=base.time_log_std,
        dist_mean=base.dist_mean, dist_std=base.dist_std,
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENCIA: caché del grafo y del modelo  (evita reprocesar desde 0)
# ─────────────────────────────────────────────────────────────────────────────
 
# Campos de GraphData que se serializan (tensores + tipos básicos picklables).
_GRAPHDATA_FIELDS = [
    'node_ids', 'node_feats', 'node_modes', 'city_ids',
    'edge_src', 'edge_dst', 'edge_feats', 'edge_times',
    'pred_src', 'pred_dst', 'pred_feats', 'adj',
    'node_meta', 'labeled_route_types', 'unlabeled_route_types',
    'n_cities', 'time_log_mean', 'time_log_std', 'dist_mean', 'dist_std',
]
 
 
def save_graphdata(data: GraphData, path: str):
    """
    Guarda un GraphData como diccionario plano (tensores + tipos básicos).
    NO serializa la clase GraphData en sí: así el archivo se puede cargar
    desde cualquier script, sin depender de qué módulo sea '__main__'.
    """
    payload = {f: getattr(data, f) for f in _GRAPHDATA_FIELDS}
    payload['_format'] = 'graphdata-dict-v1'
    torch.save(payload, path)
    print(f"Grafo cacheado → {path}")
 
 
def load_graphdata(path: str) -> GraphData:
    """
    Carga un GraphData desde caché. Acepta dos formatos:
      - el nuevo (diccionario plano), y
      - cachés antiguas que serializaban el objeto GraphData entero y que
        fallaban con "Can't get attribute 'GraphData' on __main__".
    """
    # Compatibilidad con cachés antiguas: el pickle viejo busca la clase en
    # el módulo '__main__'. Registramos un alias para que la encuentre.
    import __main__ as _main
    if not hasattr(_main, 'GraphData'):
        _main.GraphData = GraphData
 
    obj = torch.load(path, map_location='cpu', weights_only=False)
 
    if isinstance(obj, GraphData):
        data = obj                                    # objeto (formato antiguo)
    elif isinstance(obj, dict) and obj.get('_format') == 'graphdata-dict-v1':
        data = GraphData(**{k: v for k, v in obj.items() if k != '_format'})
    elif isinstance(obj, dict):
        data = GraphData(**{k: obj[k] for k in _GRAPHDATA_FIELDS if k in obj})
    else:
        raise TypeError(f"El contenido de {path} no se reconoce como GraphData.")
 
    print(f"Grafo cargado desde caché: {path}  "
          f"({len(data.node_ids)} nodos, {len(data.edge_times)} aristas)")
    return data
 
 
def load_model(model_path: str, device: str = 'cpu'):
    """
    Reconstruye un TravelTimeGNN desde un checkpoint de Trainer.save.
    Devuelve (model, norm, history). Para predecir o seguir entrenando:
        model, norm, _ = load_model('model.pt')
        trainer = Trainer(model, data, adversarial=False)
    """
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = TravelTimeGNN(**ckpt.get('config', {}))
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    print(f"Modelo cargado desde {model_path}")
    return model, ckpt['norm'], ckpt.get('history', {})
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PUNTOS DE ENTRADA: GTFS -> GraphData
# ─────────────────────────────────────────────────────────────────────────────
 
def extract_from_gtfs(path: str, city_name: Optional[str] = None,
                      labeled: bool = True, cache_path: Optional[str] = None,
                      **loader_kwargs) -> GraphData:
    """
    Construye GraphData desde UN feed GTFS.
 
    cache_path: si se indica y el archivo existe, se carga el grafo cacheado
                en vez de reparsear el GTFS; si no existe, se procesa y se
                guarda allí. Así no se reprocesa todo en cada ejecución.
    labeled=False -> todas las aristas se tratan como objetivo de predicción
    (útil para usar el feed como ciudad objetivo sin datos de tiempo).
    """
    if cache_path and os.path.exists(cache_path):
        return load_graphdata(cache_path)
    feed = gtfs_loader.load_gtfs_feed(path, city_name=city_name,
                                      labeled=labeled, **loader_kwargs)
    dm, ds, tlm, tls = _global_norms([feed])
    data = _build_graph_data(feed, city_id=0, dist_mean=dm, dist_std=ds,
                             time_log_mean=tlm, time_log_std=tls)
    if cache_path:
        save_graphdata(data, cache_path)
    return data
 
 
def extract_from_multiple_gtfs(paths: List[str],
                               city_names: Optional[List[str]] = None,
                               cache_path: Optional[str] = None,
                               **loader_kwargs) -> GraphData:
    """
    Carga y fusiona varios feeds GTFS en un GraphData multi-ciudad, con
    normalización global y city_ids para el entrenamiento adversarial.
    cache_path: igual que en extract_from_gtfs, evita reparsear los feeds.
    """
    if cache_path and os.path.exists(cache_path):
        return load_graphdata(cache_path)
    if not paths:
        raise ValueError("paths no puede estar vacío.")
    if city_names is None:
        city_names = [None] * len(paths)
 
    feeds = [gtfs_loader.load_gtfs_feed(p, city_name=c, **loader_kwargs)
             for p, c in zip(paths, city_names)]
    dm, ds, tlm, tls = _global_norms(feeds)
    graphs = [_build_graph_data(f, city_id=i, dist_mean=dm, dist_std=ds,
                                time_log_mean=tlm, time_log_std=tls)
              for i, f in enumerate(feeds)]
    del feeds                                      # liberar feeds crudos
    merged = _merge(graphs, n_cities=len(graphs))
    print(f"\nDataset combinado: {len(graphs)} ciudades | "
          f"{len(merged.node_ids)} nodos | "
          f"{merged.edge_src.numel()} aristas etiquetadas")
    if cache_path:
        save_graphdata(merged, cache_path)
    return merged
 
 
# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
 
class EdgeDataset(Dataset):
    def __init__(self, src, dst, feats, times):
        self.src, self.dst, self.feats, self.times = src, dst, feats, times
 
    def __len__(self):
        return len(self.times)
 
    def __getitem__(self, i):
        return self.src[i], self.dst[i], self.feats[i], self.times[i]
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CAPA DE MESSAGE PASSING
# ─────────────────────────────────────────────────────────────────────────────
 
class MessagePassingLayer(nn.Module):
    """h_v' = ReLU( W_self·h_v + W_neigh·mean(h_u : u∈N(v)) + b )."""
 
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim, bias=False)
        self.W_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, h, adj):
        N = h.size(0)
        if adj.size(1) > 0:
            src_idx, dst_idx = adj[0], adj[1]
            agg = torch.zeros(N, h.size(1), device=h.device)
            agg.index_add_(0, dst_idx, h[src_idx])
            deg = torch.zeros(N, device=h.device)
            deg.index_add_(0, dst_idx, torch.ones(adj.size(1), device=h.device))
            agg = agg / deg.clamp(min=1).unsqueeze(1)
        else:
            agg = torch.zeros_like(h)
        out = self.W_self(h) + self.W_neigh(agg) + self.bias
        return self.dropout(self.norm(F.relu(out)))
 
 
# ─────────────────────────────────────────────────────────────────────────────
# GRADIENT REVERSAL + DISCRIMINADOR DE CIUDAD
# ─────────────────────────────────────────────────────────────────────────────
 
class _GradReverse(torch.autograd.Function):
    """Capa identidad en forward; invierte (y escala) el gradiente en backward."""
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
 
    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.lambd, None
 
 
def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)
 
 
class CityDiscriminator(nn.Module):
    """
    Clasifica el embedding de un nodo según su ciudad de origen.
    Conectado al encoder a través del Gradient Reversal Layer: el encoder
    aprende a producir embeddings que el discriminador NO puede clasificar,
    es decir, embeddings invariantes al dominio (ciudad).
    """
    def __init__(self, node_dim, n_cities, hidden=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_cities),
        )
 
    def forward(self, h):
        return self.net(h)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# GNN DE TIEMPO DE VIAJE
# ─────────────────────────────────────────────────────────────────────────────
 
class TravelTimeGNN(nn.Module):
    """
    Encoder de nodo (features enriquecidas + one-hot de modo) -> message passing
    -> predictor de arista (MLP sobre [h_src | h_dst | edge_feats]).
    """
    def __init__(self, node_feat_dim=F_NODE, edge_in=F_EDGE,
                 node_dim=32, n_layers=2, hidden=64, dropout=0.1):
        super().__init__()
        # configuración guardada para reconstruir el modelo desde un checkpoint
        self.config = dict(node_feat_dim=node_feat_dim, edge_in=edge_in,
                           node_dim=node_dim, n_layers=n_layers,
                           hidden=hidden, dropout=dropout)
        node_in = node_feat_dim + N_MODES
        self.node_proj = nn.Sequential(
            nn.Linear(node_in, node_dim), nn.ReLU(), nn.LayerNorm(node_dim),
        )
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(node_dim, node_dim, dropout) for _ in range(n_layers)
        ])
        edge_input_dim = node_dim * 2 + edge_in
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
 
    def encode_nodes(self, node_feats, node_modes, adj):
        mode_oh = F.one_hot(node_modes, num_classes=N_MODES).float()
        x = torch.cat([node_feats, mode_oh], dim=1)
        h = self.node_proj(x)
        for layer in self.mp_layers:
            h = layer(h, adj)
        return h
 
    def predict_edges(self, h, src_idx, dst_idx, edge_feats):
        x_edge = torch.cat([h[src_idx], h[dst_idx], edge_feats], dim=1)
        return self.edge_mlp(x_edge).squeeze(1)
 
    def forward(self, node_feats, node_modes, adj, src_idx, dst_idx, edge_feats):
        h = self.encode_nodes(node_feats, node_modes, adj)
        return self.predict_edges(h, src_idx, dst_idx, edge_feats)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────────────────────────────────────
 
class Trainer:
    """
    Entrenamiento, evaluación, predicción y fine-tuning.
    Con `adversarial=True` y varias ciudades, añade el discriminador de dominio.
    """
 
    def __init__(self, model: TravelTimeGNN, data: GraphData,
                 device: str = 'cpu', lr: float = 1e-3, val_ratio: float = 0.15,
                 adversarial: bool = True, adv_weight: float = 0.1):
        self.model = model.to(device)
        self.data = data
        self.device = device
        self.adv_weight = adv_weight
        self.adversarial = adversarial and data.n_cities > 1
 
        # discriminador de ciudad (solo si tiene sentido transferir)
        params = list(model.parameters())
        if self.adversarial:
            node_dim = model.node_proj[0].out_features
            self.disc = CityDiscriminator(node_dim, data.n_cities).to(device)
            params += list(self.disc.parameters())
            print(f"Entrenamiento adversarial activo "
                  f"({data.n_cities} ciudades, peso={adv_weight}).")
        else:
            self.disc = None
 
        self.opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, patience=10, factor=0.5)
 
        # tensores fijos al device
        self.node_feats = data.node_feats.to(device)
        self.node_modes = data.node_modes.to(device)
        self.adj = data.adj.to(device)
        self.city_ids = data.city_ids.to(device)
 
        # split train / val
        n = len(data.edge_times)
        idx = list(range(n))
        random.shuffle(idx)
        split = max(int(n * (1 - val_ratio)), 1) if n else 0
        tr = torch.tensor(idx[:split], dtype=torch.long)
        va = torch.tensor(idx[split:], dtype=torch.long)
        self.train_ds = EdgeDataset(data.edge_src[tr], data.edge_dst[tr],
                                    data.edge_feats[tr], data.edge_times[tr])
        self.val_ds = EdgeDataset(data.edge_src[va], data.edge_dst[va],
                                  data.edge_feats[va], data.edge_times[va])
 
        self.history = {'train_loss': [], 'val_loss': [],
                        'val_mae_s': [], 'val_rmse_s': [], 'adv_loss': []}
 
    # ── entrenamiento ─────────────────────────────────────────────────────────
 
    def train(
        self,
        epochs=100,
        batch_size=256,
        verbose=True,
        patience=20,
        checkpoint_path=None
    ):
        loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        best_val, best_wts, no_improv = float('inf'), None, 0
 
        for epoch in range(1, epochs + 1):
            self.model.train()
            if self.disc:
                self.disc.train()
            ep_loss, ep_adv, nb = 0.0, 0.0, 0
            # rampa del peso adversarial (esquema DANN): de 0 a 1
            lambd = 2.0 / (1.0 + math.exp(-5.0 * epoch / epochs)) - 1.0
 
            for src, dst, feats, times in loader:
                src, dst = src.to(self.device), dst.to(self.device)
                feats, times = feats.to(self.device), times.to(self.device)
 
                self.opt.zero_grad()
                # encode una sola vez; se reutiliza para arista y discriminador
                h = self.model.encode_nodes(self.node_feats, self.node_modes, self.adj)
                pred = self.model.predict_edges(h, src, dst, feats)
                edge_loss = F.mse_loss(pred, times)
 
                loss = edge_loss
                if self.disc:
                    logits = self.disc(grad_reverse(h, lambd))
                    adv_loss = F.cross_entropy(logits, self.city_ids)
                    loss = loss + self.adv_weight * adv_loss
                    ep_adv += adv_loss.item()
 
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                ep_loss += edge_loss.item()
                nb += 1
 
            train_loss = ep_loss / max(nb, 1)
            val_loss, mae_s, rmse_s = self._evaluate()
            self.sched.step(val_loss)
 
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae_s'].append(mae_s)
            self.history['val_rmse_s'].append(rmse_s)
            self.history['adv_loss'].append(ep_adv / max(nb, 1))
 
            if val_loss < best_val:
                best_val = val_loss
                best_wts = {k: v.clone() for k, v in self.model.state_dict().items()}
                no_improv = 0
                # checkpoint en disco: si el entrenamiento se corta no se pierde
                if checkpoint_path:
                    self.save(checkpoint_path, quiet=True)
            else:
                no_improv += 1
 
            if verbose and epoch % 10 == 0:
                adv_txt = (f"  adv={self.history['adv_loss'][-1]:.3f}"
                           if self.disc else "")
                print(f"Época {epoch:4d} | loss_train={train_loss:.4f}  "
                      f"loss_val={val_loss:.4f}  MAE={mae_s:.1f}s  "
                      f"RMSE={rmse_s:.1f}s{adv_txt}")
 
            if no_improv >= patience:
                if verbose:
                    print(f"Early stopping en época {epoch}")
                break
 
        if best_wts:
            self.model.load_state_dict(best_wts)
        if checkpoint_path:
            self.save(checkpoint_path)
        if verbose and self.history['val_mae_s']:
            print(f"\nEntrenamiento finalizado. "
                  f"Mejor val MAE = {min(self.history['val_mae_s']):.1f}s  "
                  f"Mejor val RMSE = {min(self.history['val_rmse_s']):.1f}s")
 
    def _invert(self, t_norm):
        """log-tiempo normalizado -> segundos."""
        return torch.exp(t_norm * self.data.time_log_std + self.data.time_log_mean)
 
    def _encode(self):
        """Embeddings de todos los nodos, una sola vez y sin gradiente."""
        self.model.eval()
        with torch.no_grad():
            return self.model.encode_nodes(self.node_feats, self.node_modes, self.adj)
 
    def _predict_chunked(self, h, src, dst, feats, chunk=8192):
        """
        Predice tiempos normalizados por bloques de aristas. Los nodos ya
        vienen codificados en `h` (una sola vez); recorrer las aristas por
        trozos acota el pico de memoria en grafos grandes.
        """
        out = []
        with torch.no_grad():
            for i in range(0, len(src), chunk):
                sl = slice(i, i + chunk)
                p = self.model.predict_edges(
                    h, src[sl].to(self.device), dst[sl].to(self.device),
                    feats[sl].to(self.device))
                out.append(p.cpu())
        return torch.cat(out) if out else torch.zeros(0)
 
    def _evaluate(self):
        if len(self.val_ds) == 0:
            return 0.0, 0.0, 0.0
        h = self._encode()
        preds = self._predict_chunked(h, self.val_ds.src, self.val_ds.dst,
                                      self.val_ds.feats)
        targets = self.val_ds.times
        loss_norm = F.mse_loss(preds, targets).item()
        preds_s = self._invert(preds)
        targets_s = self._invert(targets)
        mae_s = (preds_s - targets_s).abs().mean().item()
        rmse_s = ((preds_s - targets_s) ** 2).mean().sqrt().item()
        return loss_norm, mae_s, rmse_s
 
    # ── predicción ────────────────────────────────────────────────────────────
 
    def predict_all(self, chunk=8192) -> list:
        """Predice tiempos para todas las aristas sin etiqueta (por bloques)."""
        if len(self.data.pred_src) == 0:
            print("No hay aristas sin datos para predecir.")
            return []
        h = self._encode()
        pred_norm = self._predict_chunked(h, self.data.pred_src,
                                          self.data.pred_dst,
                                          self.data.pred_feats, chunk)
        pred_s = self._invert(pred_norm)
 
        results = []
        for i, (si, di) in enumerate(zip(self.data.pred_src.tolist(),
                                         self.data.pred_dst.tolist())):
            results.append(self._make_result(si, di, float(pred_s[i]),
                                              self.data.unlabeled_route_types[i]
                                              if i < len(self.data.unlabeled_route_types)
                                              else 3))
        results.sort(key=lambda r: r.time_seconds)
        return results
 
    def predict_pair(self, src_id, dst_id) -> Optional[PredictionResult]:
        """Predice el tiempo entre dos paradas concretas."""
        idx = {nid: i for i, nid in enumerate(self.data.node_ids)}
        si, di = idx.get(src_id), idx.get(dst_id)
        if si is None or di is None:
            print(f"Parada no encontrada: {src_id if si is None else dst_id!r}")
            return None
        ms = self.data.node_meta.get(src_id, {})
        md = self.data.node_meta.get(dst_id, {})
        dist = haversine(ms.get('lat', 0.0), ms.get('lon', 0.0),
                         md.get('lat', 0.0), md.get('lon', 0.0))
        rt = int(self.data.node_modes[si].item())
        # par O-D arbitrario: se estima el nº de paradas a partir de la distancia
        n_hops = _estimate_n_hops(dist)
        feats = torch.tensor([_edge_feature_vector(
            dist, rt, False, 12.0, 1, n_hops, 0,
            self.data.dist_mean, self.data.dist_std)],
            dtype=torch.float32, device=self.device)
        h = self._encode()
        with torch.no_grad():
            pn = self.model.predict_edges(
                h, torch.tensor([si], device=self.device),
                torch.tensor([di], device=self.device), feats)
        t_s = float(self._invert(pn.cpu())[0])
        return self._make_result(si, di, t_s, rt)
 
    def _make_result(self, si, di, t_s, route_type):
        nid = self.data.node_ids
        sid, did = nid[si], nid[di]
        ms = self.data.node_meta.get(sid, {})
        md = self.data.node_meta.get(did, {})
        dist = haversine(ms.get('lat', 0.0), ms.get('lon', 0.0),
                         md.get('lat', 0.0), md.get('lon', 0.0))
        return PredictionResult(
            src_id=sid, src_name=ms.get('name', sid),
            src_lat=ms.get('lat', 0.0), src_lon=ms.get('lon', 0.0),
            src_modes=ms.get('modes', [3]),
            dst_id=did, dst_name=md.get('name', did),
            dst_lat=md.get('lat', 0.0), dst_lon=md.get('lon', 0.0),
            dst_modes=md.get('modes', [3]),
            time_seconds=t_s, time_minutes=t_s / 60,
            route_type=route_type,
            route_name=ROUTE_TYPE_NAMES.get(route_type, str(route_type)),
            distance_m=dist)
 
    # ── fine-tuning para transferencia ────────────────────────────────────────
 
    def fine_tune(self, target_data: GraphData, epochs=30, lr=2e-4,
                  freeze_mp=True, checkpoint_path=None) -> 'Trainer':
        """
        Adapta el modelo entrenado a una ciudad objetivo con pocos datos.
        freeze_mp=True congela encoder + message passing y solo reajusta el
        predictor de arista (recomendado con datos escasos).
        checkpoint_path guarda el modelo ajustado durante y al final del proceso.
        """
        if freeze_mp:
            for p in self.model.node_proj.parameters():
                p.requires_grad = False
            for layer in self.model.mp_layers:
                for p in layer.parameters():
                    p.requires_grad = False
            print("Capas de encoder/message passing congeladas.")
 
        # el fine-tuning es mono-ciudad -> sin adversarial
        ft = Trainer(self.model, target_data, device=self.device, lr=lr,
                     adversarial=False)
        ft.opt = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr, weight_decay=1e-4)
        print(f"Fine-tuning sobre {len(target_data.edge_times)} aristas "
              f"de la ciudad objetivo...")
        ft.train(epochs=epochs, verbose=True, patience=epochs,
                 checkpoint_path=checkpoint_path)
 
        for p in self.model.parameters():
            p.requires_grad = True
        return ft
 
    # ── guardado ──────────────────────────────────────────────────────────────
 
    def save(self, path, quiet=False):
        torch.save({'model_state': self.model.state_dict(),
                    'config': self.model.config,
                    'norm': {'time_log_mean': self.data.time_log_mean,
                             'time_log_std': self.data.time_log_std,
                             'dist_mean': self.data.dist_mean,
                             'dist_std': self.data.dist_std},
                    'history': self.history}, path)
        if not quiet:
            print(f"Modelo guardado → {path}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PREDICTOR POR DISTANCIA (ciudad objetivo sin grafo)
# ─────────────────────────────────────────────────────────────────────────────
 
class DistancePredictor:
    """
    Estima tiempos en una ciudad sin datos GTFS usando un modelo entrenado en
    ciudades fuente. Omite el message passing (requiere topología) y consulta
    el MLP de arista con un embedding de nodo neutro (media del espacio latente).
    El entrenamiento adversarial hace que ese embedding sea genuinamente
    invariante al dominio, no un simple promedio.
    """
    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self.model = trainer.model
        self.data = trainer.data
        self.device = trainer.device
        self.model.eval()
        with torch.no_grad():
            h = self.model.encode_nodes(self.data.node_feats.to(self.device),
                                        self.data.node_modes.to(self.device),
                                        self.data.adj.to(self.device))
            self._neutral = h.mean(dim=0, keepdim=True)
 
    def predict(self, distance_m, route_type=3, dep_hour=12.0, is_transfer=False,
                n_hops=None, n_transfers=0,
                src_name="Origen", dst_name="Destino",
                src_lat=0.0, src_lon=0.0, dst_lat=0.0, dst_lon=0.0):
        if distance_m <= 0:
            raise ValueError("distance_m debe ser positivo.")
        # si no se indica el nº de paradas, se estima a partir de la distancia
        if n_hops is None:
            n_hops = _estimate_n_hops(distance_m)
        feat = torch.tensor([_edge_feature_vector(
            distance_m, route_type, is_transfer, dep_hour, 1,
            n_hops, n_transfers, self.data.dist_mean, self.data.dist_std)],
            dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            x = torch.cat([self._neutral, self._neutral, feat], dim=1)
            pn = self.model.edge_mlp(x).squeeze()
        t_s = float(torch.exp(pn * self.data.time_log_std +
                              self.data.time_log_mean))
        return PredictionResult(
            src_id="dist", src_name=src_name, src_lat=src_lat, src_lon=src_lon,
            src_modes=[route_type], dst_id="dist", dst_name=dst_name,
            dst_lat=dst_lat, dst_lon=dst_lon, dst_modes=[route_type],
            time_seconds=t_s, time_minutes=t_s / 60, route_type=route_type,
            route_name=ROUTE_TYPE_NAMES.get(route_type, str(route_type)),
            distance_m=distance_m)
 
    def compare_modes(self, distance_m, route_types=None):
        if route_types is None:
            present = set(self.data.labeled_route_types +
                          self.data.unlabeled_route_types)
            route_types = sorted(present) if present else [1, 2, 3]
        results = sorted((self.predict(distance_m, route_type=rt)
                          for rt in route_types),
                         key=lambda r: r.time_seconds)
        print(f"\n{'─'*58}")
        print(f"  Comparación de modos — distancia: {distance_m:,.0f} m")
        print(f"{'─'*58}")
        worst = max((r.time_minutes for r in results), default=1)
        for r in results:
            bar = "█" * int(r.time_minutes / max(worst, 1) * 20)
            print(f"  {r.route_name:<14} {r.time_minutes:5.1f} min  {bar}")
        print(f"{'─'*58}")
        return results
 
 
# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE SALIDA
# ─────────────────────────────────────────────────────────────────────────────
 
def print_metrics(trainer: Trainer):
    h = trainer.history
    if not h['val_mae_s']:
        print("No hay historial de entrenamiento.")
        return
    best = int(np.argmin(h['val_mae_s']))
    print("\n── Resumen de entrenamiento ─────────────────────────────")
    print(f"  Épocas completadas : {len(h['train_loss'])}")
    print(f"  Mejor época (MAE)  : {best + 1}")
    print(f"  Mejor val MAE      : {h['val_mae_s'][best]:.1f} s "
          f"({h['val_mae_s'][best]/60:.2f} min)")
    print(f"  Mejor val RMSE     : {h['val_rmse_s'][best]:.1f} s "
          f"({h['val_rmse_s'][best]/60:.2f} min)")
    print("─────────────────────────────────────────────────────────")
 
 
def export_predictions_to_json(trainer: Trainer, output_path='predicted_times.json'):
    preds = trainer.predict_all()
    result = {'meta': {'model': 'TravelTimeGNN', 'n_predicted': len(preds),
                       'time_unit': 'seconds'},
              'predictions': [r.to_dict() for r in preds]}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Predicciones exportadas → {output_path}  ({len(preds)} aristas)")
    return result
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PIPELINES
# ─────────────────────────────────────────────────────────────────────────────
 
def run_gtfs_pipeline(gtfs_path, city_name=None, epochs=150, device='auto',
                      save_path='model.pt', pred_path='predicted_times.json',
                      graph_cache='graph_cache.pt',
                      node_dim=32, n_layers=2, hidden=64,
                      **loader_kwargs):
    """
    Pipeline de una sola ciudad desde un feed GTFS.
 
    graph_cache: ruta de caché del grafo parseado (evita reprocesar el GTFS).
    save_path:   el modelo se va guardando DURANTE el entrenamiento (mejor
                 época) y al final, para no perder el progreso ante un corte.
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo: {device}")
    print(f"\n[1/3] Cargando GTFS desde {gtfs_path}…")
    data = extract_from_gtfs(gtfs_path, city_name=city_name,
                             cache_path=graph_cache, **loader_kwargs)
    print(f"  Nodos: {len(data.node_ids)} | "
          f"Aristas etiquetadas: {len(data.edge_times)} | "
          f"A predecir: {len(data.pred_src)}")
 
    print("\n[2/3] Entrenando modelo…")
    model = TravelTimeGNN(node_dim=node_dim, n_layers=n_layers, hidden=hidden)
    trainer = Trainer(model, data, device=device, adversarial=False)
    trainer.train(epochs=epochs, verbose=True, checkpoint_path=save_path)
 
    print("\n[3/3] Evaluando y exportando…")
    print_metrics(trainer)
    if pred_path and len(data.pred_src) > 0:
        export_predictions_to_json(trainer, pred_path)
    return trainer
 
 
def run_multi_city_gtfs_pipeline(
    source_paths,
    target_path=None,
    source_names=None,
    target_name=None,
    epochs=150,
    ft_epochs=40,
    device='auto',
    adv_weight=0.1,
    freeze_mp=True,
    save_path='model_multicity.pt',
    pred_path='predicted_times.json',
    source_cache='graph_sources.pt',
    target_cache='graph_target.pt',
    node_dim=32,
    n_layers=3,
    hidden=64,
    **loader_kwargs
):
    """
    Pipeline de transferencia entre ciudades desde feeds GTFS:
      1. Fusiona ciudades fuente y entrena con adversarial domain-invariant.
      2. (Opcional) fine-tuning sobre la ciudad objetivo.
      3. Devuelve también un DistancePredictor para consultas sin grafo.
 
    source_cache / target_cache: cachean los grafos parseados.
    save_path: el modelo se guarda durante el entrenamiento y al terminar;
               el modelo ajustado se guarda en <save_path>_finetuned.pt.
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo: {device}")
 
    print(f"\n[1/3] Fusionando {len(source_paths)} ciudades fuente…")
    combined = extract_from_multiple_gtfs(source_paths, city_names=source_names,
                                          cache_path=source_cache, **loader_kwargs)
 
    print("\n[2/3] Entrenando modelo multi-ciudad (adversarial)…")
    model = TravelTimeGNN(node_dim=node_dim, n_layers=n_layers, hidden=hidden)
    trainer = Trainer(model, combined, device=device,
                      adversarial=True, adv_weight=adv_weight)
    trainer.train(epochs=epochs, verbose=True, checkpoint_path=save_path)
    print_metrics(trainer)
 
    if target_path:
        print(f"\n[3/3] Transferencia a ciudad objetivo: {target_path}")
        # la ciudad objetivo se carga con sus datos disponibles
        target = extract_from_gtfs(target_path, city_name=target_name,
                                   cache_path=target_cache, **loader_kwargs)
        if len(target.edge_times) > 0:
            ft_ckpt = (save_path.replace('.pt', '_finetuned.pt')
                       if save_path else None)
            trainer = trainer.fine_tune(target, epochs=ft_epochs,
                                        freeze_mp=freeze_mp,
                                        checkpoint_path=ft_ckpt)
        else:
            print("  Sin aristas etiquetadas en la ciudad objetivo: "
                  "se usa el modelo fuente directamente.")
        if pred_path and len(target.pred_src) > 0:
            tgt_trainer = Trainer(model, target, device=device, adversarial=False)
            export_predictions_to_json(tgt_trainer, pred_path)
 
    dp = DistancePredictor(trainer)
    print("\nDistancePredictor listo para consultas sin grafo.")
    return trainer, dp
 
 
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
 
    if len(args) >= 1 and args[0] == '--demo':
        # demostración end-to-end con feeds GTFS sintéticos
        print("Generando feeds GTFS sintéticos (3 ciudades fuente + 1 objetivo)…")
        srcs = []
        for i, name in enumerate(['Casablanca', 'Berlin', 'Sydney']):
            d = gtfs_loader.write_synthetic_feed(
                f'/tmp/_gtfs_{name}', name, n_stops=14, seed=i,
                center=(33 + i * 3, -7 + i * 5))
            srcs.append(d)
        tgt = gtfs_loader.write_synthetic_feed(
            '/tmp/_gtfs_Lima', 'Lima', n_stops=12, seed=99, center=(-12.0, -77.0))
 
        trainer, dp = run_multi_city_gtfs_pipeline(
            srcs, target_path=tgt, source_names=['Casablanca', 'Berlin', 'Sydney'],
            target_name='Lima', epochs=60, ft_epochs=25,
            save_path=None, pred_path=None,
            source_cache=None, target_cache=None, verbose=False)
 
        print("\n── Estimaciones por distancia (modelo transferido) ──")
        for km in [1, 5, 10, 20]:
            r = dp.predict(km * 1000, route_type=3)
            print(f"  Bus  {km:2d} km → {r.time_minutes:.1f} min")
        dp.compare_modes(distance_m=8000)
 
    elif len(args) == 1:
        run_gtfs_pipeline(args[0])
 
    elif len(args) >= 2 and args[0] == '--multi':
        json_args = [a for a in args[1:] if not a.startswith('--')]
        ti = args.index('--target') if '--target' in args else None
        target = args[ti + 1] if ti is not None else None
        sources = [j for j in json_args if j != target]
        run_multi_city_gtfs_pipeline(sources, target_path=target)
 
    else:
        print("Uso:")
        print("  python gnn_travel_time.py --demo")
        print("  python gnn_travel_time.py feed.zip")
        print("  python gnn_travel_time.py --multi c1.zip c2.zip [--target obj.zip]")