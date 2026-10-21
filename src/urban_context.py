"""
Contexto urbano de OSM para las celdas de la rejilla: puntos de interés (POI) y
red vial (intersecciones, semáforos, jerarquía de vías).

Siguiendo a TransferTraj, cada etiqueta OSM se proyecta sobre una taxonomía fija
y gruesa, y el entorno se describe con proporciones y rangos percentil dentro de
la ciudad, nunca con conteos absolutos: un conteo absoluto identifica a la ciudad
y estorba la transferencia entre ciudades.

Las capas son los GeoJSON de puntos que genera fetch_osm.py.
"""

import json
import math
import os
from collections import defaultdict

import numpy as np

import compat


POI_CLASSES = [
    'comercio', 'alimentacion', 'oficina', 'educacion', 'salud', 'ocio',
    'transporte', 'turismo', 'industria', 'verde', 'servicios', 'residencial',
]
POI_CLASS_IDX = {c: i for i, c in enumerate(POI_CLASSES)}
N_POI_CLASSES = len(POI_CLASSES)

# se consulta por valor (shop=supermarket) y por clave (shop=*)
OSM_TO_CLASS = {
    'shop': 'comercio', 'marketplace': 'comercio', 'mall': 'comercio',
    'supermarket': 'comercio', 'department_store': 'comercio',
    'convenience': 'comercio', 'commercial': 'comercio', 'retail': 'comercio',
    'restaurant': 'alimentacion', 'cafe': 'alimentacion', 'fast_food': 'alimentacion',
    'bar': 'alimentacion', 'pub': 'alimentacion', 'food_court': 'alimentacion',
    'office': 'oficina', 'coworking_space': 'oficina', 'bank': 'oficina',
    'government': 'oficina', 'townhall': 'oficina',
    'school': 'educacion', 'university': 'educacion', 'college': 'educacion',
    'kindergarten': 'educacion', 'library': 'educacion', 'education': 'educacion',
    'hospital': 'salud', 'clinic': 'salud', 'doctors': 'salud',
    'pharmacy': 'salud', 'dentist': 'salud', 'healthcare': 'salud',
    'cinema': 'ocio', 'theatre': 'ocio', 'nightclub': 'ocio',
    'stadium': 'ocio', 'sports_centre': 'ocio', 'fitness_centre': 'ocio',
    'leisure': 'ocio',
    'bus_station': 'transporte', 'station': 'transporte', 'parking': 'transporte',
    'fuel': 'transporte', 'taxi': 'transporte', 'ferry_terminal': 'transporte',
    'aerodrome': 'transporte',
    'hotel': 'turismo', 'museum': 'turismo', 'attraction': 'turismo',
    'viewpoint': 'turismo', 'tourism': 'turismo', 'guest_house': 'turismo',
    'industrial': 'industria', 'warehouse': 'industria', 'factory': 'industria',
    'works': 'industria', 'depot': 'industria',
    'park': 'verde', 'garden': 'verde', 'forest': 'verde', 'pitch': 'verde',
    'grass': 'verde', 'cemetery': 'verde', 'recreation_ground': 'verde',
    'police': 'servicios', 'fire_station': 'servicios', 'post_office': 'servicios',
    'place_of_worship': 'servicios', 'community_centre': 'servicios',
    'toilets': 'servicios', 'amenity': 'servicios',
    'residential': 'residencial', 'apartments': 'residencial', 'house': 'residencial',
    'dormitory': 'residencial', 'building': 'residencial',
}

ROAD_CLASSES = ['arterial', 'colectora', 'local', 'peatonal']
N_ROAD_CLASSES = len(ROAD_CLASSES)

OSM_HIGHWAY_TO_CLASS = {
    'motorway': 'arterial', 'motorway_link': 'arterial', 'trunk': 'arterial',
    'trunk_link': 'arterial', 'primary': 'arterial', 'primary_link': 'arterial',
    'secondary': 'colectora', 'secondary_link': 'colectora',
    'tertiary': 'colectora', 'tertiary_link': 'colectora',
    'residential': 'local', 'unclassified': 'local', 'living_street': 'local',
    'service': 'local', 'road': 'local',
    'pedestrian': 'peatonal', 'footway': 'peatonal', 'path': 'peatonal',
    'steps': 'peatonal', 'cycleway': 'peatonal',
}
# cuánto ralentiza cada clase de vía; resume la mezcla vial en un escalar
ROAD_FRICTION = {'arterial': 0.15, 'colectora': 0.45, 'local': 0.80, 'peatonal': 1.0}

NODE_POI_NAMES = [f'poi_{c}' for c in POI_CLASSES] + ['poi_entropy', 'poi_density_q']
NODE_ROAD_NAMES = ['inter_density_q', 'signal_ratio', 'street_count_norm',
                   'road_hierarchy'] + [f'via_{c}' for c in ROAD_CLASSES]
EDGE_CTX_NAMES = (['inter_per_km', 'signal_per_km', 'poi_corridor_q', 'friction']
                  + [f'corr_{c}' for c in ROAD_CLASSES])


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6_371_000.0 * 2 * math.asin(math.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def quantile_rank(values):
    """Rango percentil en [0, 1]. Conserva solo el orden dentro de la ciudad."""
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return a
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[a.argsort(kind='stable')] = np.arange(a.size, dtype=np.float64)
    return ranks / max(a.size - 1, 1)


def _shannon(p):
    """Entropía normalizada a [0, 1]."""
    p = np.asarray(p, dtype=np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p[p > 0] / s
    if p.size <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / math.log(len(POI_CLASSES)))


class _GridIndex:
    """Índice espacial por cubetas de tamaño fijo, para consultas de radio corto."""

    def __init__(self, points, cell_m=500.0):
        self.cell_m = cell_m
        self.points = points
        self.cells = defaultdict(list)
        if not points:
            self.lat0 = 0.0
            self._dlat = self._dlon = 1.0
            return
        self.lat0 = sum(p['lat'] for p in points) / len(points)
        self._dlat = cell_m / 110_540.0
        self._dlon = cell_m / max(111_320.0 * math.cos(math.radians(self.lat0)), 1.0)
        for i, p in enumerate(points):
            self.cells[self._key(p['lat'], p['lon'])].append(i)

    def _key(self, lat, lon):
        return (int(math.floor(lat / self._dlat)), int(math.floor(lon / self._dlon)))

    def query(self, lat, lon, radius_m):
        if not self.points:
            return []
        span = int(math.ceil(radius_m / self.cell_m))
        ci, cj = self._key(lat, lon)
        out = []
        for i in range(ci - span, ci + span + 1):
            for j in range(cj - span, cj + span + 1):
                for k in self.cells.get((i, j), ()):
                    p = self.points[k]
                    if haversine(lat, lon, p['lat'], p['lon']) <= radius_m:
                        out.append(k)
        return out


def _classify_poi(props):
    """Clase canónica por el valor de la etiqueta (shop=supermarket) o por la clave (shop=*)."""
    for key in ('amenity', 'shop', 'office', 'leisure', 'tourism', 'landuse', 'healthcare'):
        v = props.get(key)
        if not v:
            continue
        v = str(v).strip().lower()
        if v in OSM_TO_CLASS:
            return OSM_TO_CLASS[v]
        if key in OSM_TO_CLASS:
            return OSM_TO_CLASS[key]
    return None


def _read_points(path):
    """Puntos de un GeoJSON de fetch_osm.py como dicts con lat, lon y propiedades."""
    path = compat.as_path(path)
    if not os.path.exists(path):
        raise ValueError(f"Capa de contexto urbano no encontrada: {path!r}")
    with compat.open_text(path) as f:
        features = json.load(f)['features']
    rows = []
    for ft in features:
        geom = ft.get('geometry') or {}
        if geom.get('type') == 'Point':
            lon, lat = geom['coordinates'][:2]
            rows.append({**(ft.get('properties') or {}), 'lat': float(lat), 'lon': float(lon)})
    return rows


def load_pois(path):
    out = []
    for r in _read_points(path):
        cls = _classify_poi(r)
        if cls is not None:
            out.append({'lat': r['lat'], 'lon': r['lon'], 'cls': cls})
    return out


def load_road_nodes(path):
    """Nodos viales; street_count >= 3 es una intersección real."""
    return [{'lat': r['lat'], 'lon': r['lon'],
             'street_count': int(r.get('street_count', 3)),
             'signal': bool(r.get('signal')),
             'cls': OSM_HIGHWAY_TO_CLASS.get(str(r.get('highway', '')).lower(), 'local')}
            for r in _read_points(path)]


class UrbanContext:
    """Capas POI y vial de una ciudad para consultas por celda."""

    def __init__(self, pois=None, road_nodes=None, *, poi_radius_m=400.0,
                 city=None, verbose=True):
        self.city = city
        self.poi_radius_m = poi_radius_m
        self.pois = load_pois(pois) if pois is not None else []
        self.roads = load_road_nodes(road_nodes) if road_nodes is not None else []
        self.has_poi = len(self.pois) > 0
        self.has_road = len(self.roads) > 0
        self._poi_idx = _GridIndex(self.pois, cell_m=max(poi_radius_m, 250.0))
        self._road_idx = _GridIndex(self.roads, cell_m=250.0)
        if verbose:
            print(f"[ctx:{city or '?'}] {len(self.pois)} POIs | "
                  f"{len(self.roads)} nodos viales")

    def annotate_nodes(self, nodes):
        """Añade a cada celda las claves de NODE_POI_NAMES y NODE_ROAD_NAMES.

        Necesita todas las celdas a la vez porque las densidades se pasan a
        rango percentil dentro de la ciudad.
        """
        n = len(nodes)
        mix = np.zeros((n, N_POI_CLASSES))
        poi_count = np.zeros(n)
        inter_count = np.zeros(n)
        signal_frac = np.zeros(n)
        street_mean = np.zeros(n)
        road_mix = np.zeros((n, N_ROAD_CLASSES))

        for i, nd in enumerate(nodes):
            lat, lon = nd['lat'], nd['lon']
            if self.has_poi:
                for k in self._poi_idx.query(lat, lon, self.poi_radius_m):
                    mix[i, POI_CLASS_IDX[self.pois[k]['cls']]] += 1.0
                poi_count[i] = mix[i].sum()
            if self.has_road:
                inter = sig = sc = 0
                for k in self._road_idx.query(lat, lon, self.poi_radius_m):
                    r = self.roads[k]
                    if r['street_count'] >= 3:
                        inter += 1
                        sc += r['street_count']
                        sig += bool(r['signal'])
                    road_mix[i, ROAD_CLASSES.index(r['cls'])] += 1.0
                inter_count[i] = inter
                signal_frac[i] = sig / max(inter, 1)
                street_mean[i] = (sc / max(inter, 1)) / 8.0

        mix_p = mix / np.maximum(mix.sum(axis=1, keepdims=True), 1.0)
        road_p = road_mix / np.maximum(road_mix.sum(axis=1, keepdims=True), 1.0)
        poi_q = quantile_rank(poi_count) if self.has_poi else np.zeros(n)
        inter_q = quantile_rank(inter_count) if self.has_road else np.zeros(n)
        hierarchy = road_p @ np.array([ROAD_FRICTION[c] for c in ROAD_CLASSES])

        for i, nd in enumerate(nodes):
            for c, v in zip(POI_CLASSES, mix_p[i]):
                nd[f'poi_{c}'] = float(v)
            nd['poi_entropy'] = _shannon(mix[i])
            nd['poi_density_q'] = float(poi_q[i])
            nd['inter_density_q'] = float(inter_q[i])
            nd['signal_ratio'] = float(signal_frac[i])
            nd['street_count_norm'] = float(street_mean[i])
            nd['road_hierarchy'] = float(hierarchy[i])
            for c, v in zip(ROAD_CLASSES, road_p[i]):
                nd[f'via_{c}'] = float(v)
            nd['has_poi_ctx'] = self.has_poi
            nd['has_road_ctx'] = self.has_road


def finalize_edge_context(trips):
    """Pasa la densidad cruda de POI del corredor a rango percentil de la ciudad."""
    with_ctx = [t for t in trips if '_poi_raw' in t]
    if not with_ctx:
        return
    for t, v in zip(with_ctx, quantile_rank([t['_poi_raw'] for t in with_ctx])):
        t['poi_corridor_q'] = float(v)
        del t['_poi_raw']


def empty_edge_context():
    return dict.fromkeys(EDGE_CTX_NAMES, 0.0)
