"""
urban_context.py
================
Enriquece un feed GTFS con dos modalidades espaciales adicionales:

    POI          puntos de interés alrededor de cada parada y a lo largo del
                 corredor de cada tramo.
    RED VIAL     intersecciones, semáforos, giros y jerarquía de vías que el
                 vehículo atraviesa entre dos paradas.

Diseño (y por qué es así)
-------------------------
La referencia directa es TransferTraj (NeurIPS 2025), que representa cada punto
de trayectoria con cuatro modalidades —espacial, temporal, POI y red vial— y
demuestra en su ablación que quitar la modalidad POI o la de red vial degrada
las tres tareas. La diferencia respecto a ese trabajo es que aquí NO se usa un
codificador de texto preentrenado sobre la descripción del POI: eso exigiría un
modelo externo y haría el pipeline dependiente de una API. En su lugar se
proyecta cada etiqueta OSM sobre una TAXONOMÍA CANÓNICA fija (POI_CLASSES) y se
describe el entorno por su COMPOSICIÓN (proporciones que suman 1), no por
conteos. El motivo es el mismo que persigue TransferTraj con el texto: obtener
semántica independiente de la región.

Regla de oro de todo este módulo: NINGUNA feature absoluta.
  - Las composiciones son proporciones (suman 1).
  - Las densidades se expresan como RANGO PERCENTIL DENTRO DE LA CIUDAD.
  - Las magnitudes de corredor se normalizan POR KILÓMETRO.
Un conteo absoluto de POIs es una huella dactilar casi perfecta de la ciudad:
el discriminador de dominio con Gradient Reversal la borraría (desperdiciando
la feature) o la usaría como atajo (transferencia negativa). Ver la discusión
de "shortcut features" en OOD4TTE (KDD 2026).

Fuentes de datos admitidas (sin dependencias externas)
------------------------------------------------------
    - GeoJSON  (FeatureCollection de Point / LineString)
    - CSV      (columnas lat/lon + una columna de etiqueta)
    - listas de dicts en memoria
Para descargar de OSM se recomienda osmnx/Overpass fuera de este módulo y
volcar el resultado a GeoJSON; así el entrenamiento nunca depende de la red.

Dependencias: stdlib + numpy (igual que gtfs_loader).
"""

import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

import compat


# ─────────────────────────────────────────────────────────────────────────────
# TAXONOMÍA CANÓNICA
# ─────────────────────────────────────────────────────────────────────────────

# 12 clases. Deliberadamente gruesas: cuanto más fina la taxonomía, más
# dependiente es del esquema de etiquetado local y menos transferible.
POI_CLASSES = [
    'comercio', 'alimentacion', 'oficina', 'educacion', 'salud', 'ocio',
    'transporte', 'turismo', 'industria', 'verde', 'servicios', 'residencial',
]
POI_CLASS_IDX = {c: i for i, c in enumerate(POI_CLASSES)}
N_POI_CLASSES = len(POI_CLASSES)

# Mapeo etiqueta OSM -> clase canónica. Se consulta por 'valor' y por
# 'clave=valor'. Ampliable sin tocar el resto del código.
OSM_TO_CLASS = {
    # comercio
    'shop': 'comercio', 'marketplace': 'comercio', 'mall': 'comercio',
    'supermarket': 'comercio', 'department_store': 'comercio',
    'convenience': 'comercio', 'commercial': 'comercio', 'retail': 'comercio',
    # alimentación / consumo en sitio
    'restaurant': 'alimentacion', 'cafe': 'alimentacion', 'fast_food': 'alimentacion',
    'bar': 'alimentacion', 'pub': 'alimentacion', 'food_court': 'alimentacion',
    # oficina
    'office': 'oficina', 'coworking_space': 'oficina', 'bank': 'oficina',
    'government': 'oficina', 'townhall': 'oficina',
    # educación
    'school': 'educacion', 'university': 'educacion', 'college': 'educacion',
    'kindergarten': 'educacion', 'library': 'educacion', 'education': 'educacion',
    # salud
    'hospital': 'salud', 'clinic': 'salud', 'doctors': 'salud',
    'pharmacy': 'salud', 'dentist': 'salud', 'healthcare': 'salud',
    # ocio
    'cinema': 'ocio', 'theatre': 'ocio', 'nightclub': 'ocio',
    'stadium': 'ocio', 'sports_centre': 'ocio', 'fitness_centre': 'ocio',
    'leisure': 'ocio',
    # transporte
    'bus_station': 'transporte', 'station': 'transporte', 'parking': 'transporte',
    'fuel': 'transporte', 'taxi': 'transporte', 'ferry_terminal': 'transporte',
    'aerodrome': 'transporte',
    # turismo
    'hotel': 'turismo', 'museum': 'turismo', 'attraction': 'turismo',
    'viewpoint': 'turismo', 'tourism': 'turismo', 'guest_house': 'turismo',
    # industria
    'industrial': 'industria', 'warehouse': 'industria', 'factory': 'industria',
    'works': 'industria', 'depot': 'industria',
    # áreas verdes / abiertas
    'park': 'verde', 'garden': 'verde', 'forest': 'verde', 'pitch': 'verde',
    'grass': 'verde', 'cemetery': 'verde', 'recreation_ground': 'verde',
    # servicios públicos varios
    'police': 'servicios', 'fire_station': 'servicios', 'post_office': 'servicios',
    'place_of_worship': 'servicios', 'community_centre': 'servicios',
    'toilets': 'servicios', 'amenity': 'servicios',
    # residencial
    'residential': 'residencial', 'apartments': 'residencial', 'house': 'residencial',
    'dormitory': 'residencial', 'building': 'residencial',
}

# Jerarquía vial canónica: 4 niveles. Igual que arriba, gruesa a propósito.
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
# Peso de fricción por clase: cuánto ralentiza recorrer esa jerarquía.
# Sirve para colapsar la mezcla vial en un escalar interpretable.
ROAD_FRICTION = {'arterial': 0.15, 'colectora': 0.45, 'local': 0.80, 'peatonal': 1.0}


# ─────────────────────────────────────────────────────────────────────────────
# NOMBRES DE LAS FEATURES QUE PRODUCE ESTE MÓDULO
# (gnn_travel_time los importa; no duplicar la lista allí)
# ─────────────────────────────────────────────────────────────────────────────

# Modalidad POI del NODO: composición + entropía + densidad percentil
NODE_POI_NAMES = [f'poi_{c}' for c in POI_CLASSES] + ['poi_entropy', 'poi_density_q']
# Modalidad RED VIAL del NODO
NODE_ROAD_NAMES = ['inter_density_q', 'signal_ratio', 'street_count_norm',
                   'road_hierarchy'] + [f'via_{c}' for c in ROAD_CLASSES]
# Contexto de la ARISTA (lo que se atraviesa a lo largo del corredor)
EDGE_CTX_NAMES = ['inter_per_km', 'signal_per_km', 'turn_per_km',
                  'circuity', 'poi_corridor_q'] + [f'corr_{c}' for c in ROAD_CLASSES] + \
                 ['friction']

N_NODE_POI = len(NODE_POI_NAMES)
N_NODE_ROAD = len(NODE_ROAD_NAMES)
N_EDGE_CTX = len(EDGE_CTX_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES GEOMÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

_R_EARTH = 6_371_000.0


def _haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return _R_EARTH * 2 * math.asin(math.sqrt(a))


def _bearing(lat1, lon1, lat2, lon2):
    """Rumbo en grados [0,360). Se usa para el ángulo de giro acumulado."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _turn(b1, b2):
    """Diferencia angular absoluta entre dos rumbos, en [0,180]."""
    d = abs(b2 - b1) % 360.0
    return d if d <= 180.0 else 360.0 - d


def _quantile_rank(values):
    """
    Rango percentil dentro de la ciudad, en [0,1].

    Sustituye deliberadamente al z-score para las densidades: el z-score sigue
    dependiendo de la media y la varianza de la ciudad (Lima y Berlín tienen
    escalas de densidad de POI incomparables), mientras que el rango percentil
    solo conserva el ORDEN relativo dentro de la ciudad, que sí es la señal
    transferible ("esta parada está en una zona densa PARA SU CIUDAD").
    """
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return a
    order = a.argsort(kind='stable')
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = np.arange(a.size, dtype=np.float64)
    return ranks / max(a.size - 1, 1)


def _shannon(p):
    """Entropía de Shannon normalizada a [0,1] de un vector de proporciones."""
    p = np.asarray(p, dtype=np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p[p > 0] / s
    if p.size <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / math.log(len(POI_CLASSES)))


# ─────────────────────────────────────────────────────────────────────────────
# ÍNDICE ESPACIAL POR REJILLA
# ─────────────────────────────────────────────────────────────────────────────

class _GridIndex:
    """
    Índice espacial por celdas de tamaño fijo. Alternativa sin dependencias a
    un KD-tree/STRtree: para consultas de radio pequeño (<1 km) sobre cientos
    de miles de puntos el coste es prácticamente O(1) por consulta.
    """

    def __init__(self, points, cell_m=500.0):
        self.cell_m = cell_m
        self.points = points
        self.cells = defaultdict(list)
        if not points:
            self.lat0 = 0.0
            self._dlat = self._dlon = 1.0
            return
        self.lat0 = sum(p['lat'] for p in points) / len(points)
        # grados por metro en esta latitud
        self._dlat = cell_m / 110_540.0
        self._dlon = cell_m / max(111_320.0 * math.cos(math.radians(self.lat0)), 1.0)
        for i, p in enumerate(points):
            self.cells[self._key(p['lat'], p['lon'])].append(i)

    def _key(self, lat, lon):
        return (int(math.floor(lat / self._dlat)), int(math.floor(lon / self._dlon)))

    def query(self, lat, lon, radius_m):
        """Índices de los puntos dentro del radio (metros)."""
        if not self.points:
            return []
        span = int(math.ceil(radius_m / self.cell_m))
        ci, cj = self._key(lat, lon)
        out = []
        for i in range(ci - span, ci + span + 1):
            for j in range(cj - span, cj + span + 1):
                for k in self.cells.get((i, j), ()):
                    p = self.points[k]
                    if _haversine(lat, lon, p['lat'], p['lon']) <= radius_m:
                        out.append(k)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE CAPAS
# ─────────────────────────────────────────────────────────────────────────────

def _classify_poi(props):
    """Proyecta las etiquetas OSM de un POI sobre la taxonomía canónica."""
    for key in ('poi_class', 'class', 'category'):
        v = props.get(key)
        if v in POI_CLASS_IDX:
            return v
    for key in ('amenity', 'shop', 'office', 'leisure', 'tourism', 'building',
                'landuse', 'healthcare', 'highway', 'railway'):
        v = props.get(key)
        if not v:
            continue
        v = str(v).strip().lower()
        if v in OSM_TO_CLASS:
            return OSM_TO_CLASS[v]
        if key in OSM_TO_CLASS:          # p.ej. shop=* sin valor reconocido
            return OSM_TO_CLASS[key]
    return None


def _classify_road(props):
    hw = props.get('road_class') or props.get('highway') or props.get('fclass')
    if hw is None:
        return 'local'
    hw = str(hw).strip().lower()
    if hw in ROAD_CLASSES:
        return hw
    return OSM_HIGHWAY_TO_CLASS.get(hw, 'local')


def load_pois(source):
    """
    Carga una capa de POIs. `source` puede ser:
      - ruta a .geojson / .json (FeatureCollection de Point)
      - ruta a .csv con columnas lat/lon (+ etiquetas OSM como columnas)
      - lista de dicts ya normalizados
    Devuelve list[{'lat','lon','cls'}]; descarta lo que no se pueda clasificar.
    """
    rows = _read_any(source)
    out = []
    for r in rows:
        cls = _classify_poi(r)
        if cls is None or r.get('lat') is None:
            continue
        out.append({'lat': r['lat'], 'lon': r['lon'], 'cls': cls})
    return out


def load_road_nodes(source):
    """
    Carga la capa de NODOS VIALES (intersecciones). Campos reconocidos:
      street_count  nº de vías incidentes  (>=3 => intersección real)
      signal        booleano / highway=traffic_signals
      highway       jerarquía de la vía más importante que incide

    En osmnx: `ox.graph_from_bbox(...)`; los nodos traen `street_count` y las
    intersecciones semaforizadas aparecen como highway=traffic_signals.
    Devuelve list[{'lat','lon','street_count','signal','cls'}].
    """
    rows = _read_any(source)
    out = []
    for r in rows:
        if r.get('lat') is None:
            continue
        try:
            sc = int(float(r.get('street_count', 3) or 3))
        except (TypeError, ValueError):
            sc = 3
        sig = r.get('signal')
        if sig is None:
            sig = str(r.get('highway', '')).lower() in (
                'traffic_signals', 'crossing', 'stop', 'give_way', 'mini_roundabout')
        out.append({'lat': r['lat'], 'lon': r['lon'],
                    'street_count': sc,
                    'signal': bool(sig) and str(sig).lower() not in ('0', 'false', ''),
                    'cls': _classify_road(r)})
    return out


def _read_any(source):
    """Normaliza GeoJSON / CSV / lista a una lista de dicts con lat, lon y props."""
    if isinstance(source, (list, tuple)):
        return [dict(r) for r in source]
    # os.fspath en vez de isinstance(source, str): así un pathlib.Path —que es
    # lo natural en Windows— se acepta igual que una cadena
    if not compat.is_path_like(source):
        raise TypeError(f"Fuente de contexto urbano no admitida: {type(source)}")
    source = compat.as_path(source)
    if not os.path.exists(source):
        raise ValueError(f"Fuente de contexto urbano no encontrada: {source!r}")

    ext = os.path.splitext(source)[1].lower()
    if ext in ('.geojson', '.json'):
        with compat.open_text(source) as f:
            gj = json.load(f)
        feats = gj.get('features', gj if isinstance(gj, list) else [])
        rows = []
        for ft in feats:
            geom = ft.get('geometry') or {}
            coords = geom.get('coordinates')
            gtype = geom.get('type')
            if gtype == 'Point' and coords:
                lat, lon = float(coords[1]), float(coords[0])
            elif gtype in ('LineString', 'MultiPoint') and coords:
                # centroide simple: sirve para capas de vías usadas como puntos
                lat = sum(c[1] for c in coords) / len(coords)
                lon = sum(c[0] for c in coords) / len(coords)
            else:
                continue
            row = dict(ft.get('properties') or {})
            row['lat'], row['lon'] = lat, lon
            rows.append(row)
        return rows

    with compat.open_csv(source) as f:
        rows = []
        for row in csv.DictReader(f):
            lat = row.get('lat') or row.get('latitude') or row.get('stop_lat')
            lon = row.get('lon') or row.get('lng') or row.get('longitude')
            try:
                row['lat'], row['lon'] = float(lat), float(lon)
            except (TypeError, ValueError):
                continue
            rows.append(row)
        return rows


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXTO URBANO
# ─────────────────────────────────────────────────────────────────────────────

class UrbanContext:
    """
    Capa de contexto de UNA ciudad. Se pasa a gtfs_loader.load_gtfs_feed y
    éste la consulta al construir nodos y aristas.

    Uso:
        ctx = UrbanContext(pois='lima_pois.geojson',
                           road_nodes='lima_intersections.geojson')
        feed = gtfs_loader.load_gtfs_feed('lima.zip', urban_context=ctx)

    Radios: TransferTraj usa 100 m para POI y red vial alrededor del punto de
    trayectoria. Aquí el nodo es una PARADA (no un punto GPS): el área de
    captación relevante es el radio de caminata, por eso poi_radius_m=400. El
    corredor de la arista sí se queda en 100 m, alineado con TransferTraj.
    """

    def __init__(self, pois=None, road_nodes=None, *,
                 poi_radius_m=400.0, corridor_m=100.0,
                 sample_step_m=50.0, city=None, verbose=True):
        self.city = city
        self.poi_radius_m = poi_radius_m
        self.corridor_m = corridor_m
        self.sample_step_m = sample_step_m

        self.pois = load_pois(pois) if pois is not None else []
        self.roads = load_road_nodes(road_nodes) if road_nodes is not None else []
        self.has_poi = len(self.pois) > 0
        self.has_road = len(self.roads) > 0

        self._poi_idx = _GridIndex(self.pois, cell_m=max(poi_radius_m, 250.0))
        self._road_idx = _GridIndex(self.roads, cell_m=max(corridor_m * 2, 250.0))

        # caché de perfiles por polilínea (los patrones de ruta se repiten
        # mucho entre viajes: sin caché se recalcularía miles de veces)
        self._profile_cache = {}

        if verbose:
            print(f"[ctx:{city or '?'}] {len(self.pois)} POIs | "
                  f"{len(self.roads)} nodos viales | "
                  f"radio POI {poi_radius_m:.0f} m | corredor {corridor_m:.0f} m")

    # ── modalidades de NODO ───────────────────────────────────────────────────

    def annotate_nodes(self, nodes):
        """
        Añade in-place a cada nodo las claves de NODE_POI_NAMES y
        NODE_ROAD_NAMES más las banderas 'has_poi_ctx' / 'has_road_ctx'.

        Requiere la lista COMPLETA de nodos porque las densidades se convierten
        a rango percentil dentro de la ciudad.
        """
        n = len(nodes)
        mix = np.zeros((n, N_POI_CLASSES), dtype=np.float64)
        poi_count = np.zeros(n)
        inter_count = np.zeros(n)
        signal_frac = np.zeros(n)
        street_mean = np.zeros(n)
        road_mix = np.zeros((n, N_ROAD_CLASSES), dtype=np.float64)

        for i, nd in enumerate(nodes):
            lat, lon = nd['lat'], nd['lon']

            if self.has_poi:
                for k in self._poi_idx.query(lat, lon, self.poi_radius_m):
                    mix[i, POI_CLASS_IDX[self.pois[k]['cls']]] += 1.0
                poi_count[i] = mix[i].sum()

            if self.has_road:
                idxs = self._road_idx.query(lat, lon, self.poi_radius_m)
                inter, sig, sc = 0, 0, 0
                for k in idxs:
                    r = self.roads[k]
                    if r['street_count'] >= 3:
                        inter += 1
                        sc += r['street_count']
                        if r['signal']:
                            sig += 1
                    road_mix[i, ROAD_CLASSES.index(r['cls'])] += 1.0
                inter_count[i] = inter
                signal_frac[i] = sig / max(inter, 1)
                street_mean[i] = (sc / max(inter, 1)) / 8.0     # ~[0,1]

        # composición (proporciones) + entropía
        mix_sum = mix.sum(axis=1, keepdims=True)
        mix_p = np.divide(mix, np.maximum(mix_sum, 1.0))
        road_sum = road_mix.sum(axis=1, keepdims=True)
        road_p = np.divide(road_mix, np.maximum(road_sum, 1.0))

        poi_q = _quantile_rank(poi_count) if self.has_poi else np.zeros(n)
        inter_q = _quantile_rank(inter_count) if self.has_road else np.zeros(n)
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
            nd['has_poi_ctx'] = bool(self.has_poi)
            nd['has_road_ctx'] = bool(self.has_road)

    # ── modalidad de ARISTA: perfil a lo largo del corredor ───────────────────

    def path_profile(self, polyline, cache_key=None):
        """
        Perfil ACUMULADO a lo largo de una polilínea [(lat,lon), ...].

        Devuelve un dict de arrays de longitud len(polyline) con los valores
        acumulados desde el vértice 0. Así, el contexto de CUALQUIER
        sub-recorrido (i, j) se obtiene por diferencia — que es exactamente lo
        que necesitan las aristas de trayecto O-D, sin recorrer el corredor
        otra vez por cada par muestreado.
        """
        if cache_key is not None and cache_key in self._profile_cache:
            return self._profile_cache[cache_key]

        m = len(polyline)
        prof = {
            'len': np.zeros(m), 'inter': np.zeros(m), 'signal': np.zeros(m),
            'turn': np.zeros(m), 'chord': np.zeros(m), 'poi': np.zeros(m),
            'cls': np.zeros((m, N_ROAD_CLASSES)),
        }
        if m < 2:
            if cache_key is not None:
                self._profile_cache[cache_key] = prof
            return prof

        prev_bearing = None
        for k in range(1, m):
            la1, lo1 = polyline[k - 1]
            la2, lo2 = polyline[k]
            seg_len = _haversine(la1, lo1, la2, lo2)

            n_inter = n_sig = n_poi = 0
            cls_hits = np.zeros(N_ROAD_CLASSES)
            if seg_len > 0 and (self.has_road or self.has_poi):
                # muestreo del segmento cada sample_step_m; se cuentan
                # entidades únicas para no contar dos veces la misma
                steps = max(int(seg_len // self.sample_step_m), 1)
                seen_r, seen_p = set(), set()
                for s in range(steps + 1):
                    t = s / steps
                    plat = la1 + (la2 - la1) * t
                    plon = lo1 + (lo2 - lo1) * t
                    if self.has_road:
                        for idx in self._road_idx.query(plat, plon, self.corridor_m):
                            if idx in seen_r:
                                continue
                            seen_r.add(idx)
                            r = self.roads[idx]
                            cls_hits[ROAD_CLASSES.index(r['cls'])] += 1
                            if r['street_count'] >= 3:
                                n_inter += 1
                                if r['signal']:
                                    n_sig += 1
                    if self.has_poi:
                        for idx in self._poi_idx.query(plat, plon, self.corridor_m):
                            if idx not in seen_p:
                                seen_p.add(idx)
                                n_poi += 1

            turn = 0.0
            b = _bearing(la1, lo1, la2, lo2)
            if prev_bearing is not None:
                turn = _turn(prev_bearing, b)
            prev_bearing = b

            prof['len'][k] = prof['len'][k - 1] + seg_len
            prof['inter'][k] = prof['inter'][k - 1] + n_inter
            prof['signal'][k] = prof['signal'][k - 1] + n_sig
            prof['turn'][k] = prof['turn'][k - 1] + turn
            prof['poi'][k] = prof['poi'][k - 1] + n_poi
            prof['cls'][k] = prof['cls'][k - 1] + cls_hits

        prof['pts'] = polyline
        if cache_key is not None:
            self._profile_cache[cache_key] = prof
        return prof

    def span_features(self, prof, i, j):
        """
        Features de contexto del sub-recorrido entre los vértices i y j de un
        perfil. Todo se normaliza POR KILÓMETRO o es una proporción, de modo
        que la longitud del tramo (que ya está en 'dist_norm') no se cuele
        otra vez por la puerta de atrás como atajo.
        """
        if j <= i or 'pts' not in prof:
            return dict.fromkeys(EDGE_CTX_NAMES, 0.0)

        length = float(prof['len'][j] - prof['len'][i])
        km = max(length, 1.0) / 1000.0

        pts = prof['pts']
        chord = _haversine(pts[i][0], pts[i][1], pts[j][0], pts[j][1])
        circuity = length / max(chord, 1.0) if chord > 0 else 1.0

        cls = prof['cls'][j] - prof['cls'][i]
        cls_p = cls / max(cls.sum(), 1.0)
        friction = float(cls_p @ np.array([ROAD_FRICTION[c] for c in ROAD_CLASSES]))

        out = {
            # escaladas para caer en un rango ~[0,2] sin normalizador global
            'inter_per_km': float(prof['inter'][j] - prof['inter'][i]) / km / 20.0,
            'signal_per_km': float(prof['signal'][j] - prof['signal'][i]) / km / 10.0,
            'turn_per_km': float(prof['turn'][j] - prof['turn'][i]) / km / 180.0,
            'circuity': float(min(circuity, 3.0)),
            'poi_corridor_q': 0.0,      # se rellena al percentilar la ciudad
            'friction': friction,
        }
        for c, v in zip(ROAD_CLASSES, cls_p):
            out[f'corr_{c}'] = float(v)
        # crudo, para el percentil posterior
        out['_poi_raw'] = float(prof['poi'][j] - prof['poi'][i]) / km
        return out


def finalize_edge_context(edges):
    """
    Convierte la densidad cruda de POI del corredor en rango percentil DENTRO
    de la ciudad y elimina el campo auxiliar. Se llama una vez construidas
    todas las aristas del feed.
    """
    with_ctx = [e for e in edges if '_poi_raw' in e]
    if not with_ctx:
        return
    q = _quantile_rank([e['_poi_raw'] for e in with_ctx])
    for e, v in zip(with_ctx, q):
        e['poi_corridor_q'] = float(v)
        e.pop('_poi_raw', None)


def empty_edge_context():
    """Contexto neutro para aristas sin datos (o feeds sin capa OSM)."""
    d = dict.fromkeys(EDGE_CTX_NAMES, 0.0)
    d['circuity'] = 1.0
    return d


# ─────────────────────────────────────────────────────────────────────────────
# CAPA SINTÉTICA  (solo para la demo end-to-end sin descargar OSM)
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_context(nodes, seed=0, n_pois=1200, n_inter=900, city=None):
    """
    Genera una capa plausible de POIs e intersecciones alrededor de un conjunto
    de paradas. SOLO para pruebas: los resultados obtenidos con esta capa no
    tienen validez experimental, sirven para verificar que el pipeline corre.
    """
    import random
    rng = random.Random(seed)
    lats = [n['lat'] for n in nodes]
    lons = [n['lon'] for n in nodes]
    lat0, lon0 = sum(lats) / len(lats), sum(lons) / len(lons)
    span_lat = max(max(lats) - min(lats), 0.01)
    span_lon = max(max(lons) - min(lons), 0.01)

    pois = []
    for _ in range(n_pois):
        # concentración gaussiana hacia el centro: imita el gradiente urbano
        pois.append({
            'lat': lat0 + rng.gauss(0, span_lat / 3),
            'lon': lon0 + rng.gauss(0, span_lon / 3),
            'poi_class': rng.choice(POI_CLASSES),
        })
    roads = []
    for _ in range(n_inter):
        roads.append({
            'lat': lat0 + rng.uniform(-span_lat, span_lat),
            'lon': lon0 + rng.uniform(-span_lon, span_lon),
            'street_count': rng.choice([2, 3, 3, 4, 4, 5]),
            'signal': rng.random() < 0.3,
            'highway': rng.choice(['primary', 'secondary', 'tertiary',
                                   'residential', 'residential', 'service']),
        })
    return UrbanContext(pois=pois, road_nodes=roads, city=city, verbose=False)