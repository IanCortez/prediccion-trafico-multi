"""
trajectory_loader.py
====================
Reemplaza a gtfs_loader.py. Consume el formato de trayectorias de taxi del reto
ECML/PKDD 2015 (Porto) y produce las muestras de entrenamiento y test para
estimación de tiempo de viaje origen-destino (OD TTE).

Formato de entrada (CSV):
    TRIP_ID, CALL_TYPE, ORIGIN_CALL, ORIGIN_STAND, TAXI_ID, TIMESTAMP,
    DAY_TYPE, MISSING_DATA, POLYLINE

    POLYLINE es una lista JSON de pares [lon, lat] muestreados cada 15 s.

────────────────────────────────────────────────────────────────────────────
AVISO DE FUGA DE ETIQUETA — leer antes de tocar las features
────────────────────────────────────────────────────────────────────────────
En este dataset el tiempo de viaje NO es una variable observada aparte: se
define como

        travel_time = (len(POLYLINE) - 1) * 15  segundos

Es decir, la etiqueta es una función determinista del número de puntos de la
polilínea. Por tanto CUALQUIER feature derivada de la polilínea recorrida
—su longitud, su número de vértices, las intersecciones que cruza, su
circuidad— es la etiqueta disfrazada. Un modelo con esas features alcanza
R²≈1 y no ha aprendido nada.

Por eso la polilínea completa solo se retiene con `keep_polyline=True` y jamás
entra en el vector de features. Las entradas admisibles para OD TTE son las
que un sistema real tendría ANTES de que el viaje ocurra: origen, destino,
instante de salida, metadatos de la llamada y el contexto urbano estático del
corredor recto O→D. Es también lo que hace TransferTraj en su tarea de OD TTE.

────────────────────────────────────────────────────────────────────────────
Regiones en lugar de grafo
────────────────────────────────────────────────────────────────────────────
Sin GTFS no hay paradas ni topología de servicio. La unidad espacial pasa a ser
la CELDA de una rejilla regular, que es la abstracción de "nodo" que usan tanto
TSJT (bank de prompts por nodo) como los trabajos de transferencia cross-city
en general. Las celdas se anotan con las modalidades POI y de red vial de
urban_context, exactamente igual que antes se anotaban las paradas.

Dependencias: stdlib + numpy.
"""

import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

import compat
import urban_context as urban_context_mod

# Intervalo de muestreo del dataset de Porto, en segundos.
SAMPLING_INTERVAL_S = 15.0

CALL_TYPES = ['A', 'B', 'C']       # A: central, B: parada, C: aleatorio
DAY_TYPES = ['A', 'B', 'C']        # A: normal, B: víspera de festivo, C: festivo


def _haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _bearing(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


# ─────────────────────────────────────────────────────────────────────────────
# REJILLA DE REGIONES
# ─────────────────────────────────────────────────────────────────────────────

class RegionGrid:
    """
    Rejilla regular sobre el área de la ciudad. Cada celda ocupada actúa como
    "nodo": recibe las modalidades POI / red vial y tiene su propio prompt en
    el NPM.

    El tamaño de celda es un compromiso: celdas grandes agregan bien pero
    diluyen el contexto; celdas pequeñas dan más resolución pero dejan la
    mayoría con muy pocos viajes. 500 m funciona bien para un área urbana del
    tamaño de Porto (el área de entrenamiento de TransferTraj para Porto es de
    11,13 × 5,11 km).
    """

    def __init__(self, cell_m=500.0):
        self.cell_m = cell_m
        self.lat0 = self.lon0 = 0.0
        self._dlat = self._dlon = 1.0
        self.cell_ids = []          # lista de (i, j) ocupadas
        self.cell_index = {}        # (i, j) -> posición en cell_ids
        self.cell_count = []        # nº de puntos observados por celda

    def fit(self, lats, lons):
        self.lat0, self.lon0 = float(np.min(lats)), float(np.min(lons))
        self.lat_max, self.lon_max = float(np.max(lats)), float(np.max(lons))
        mid_lat = (self.lat0 + self.lat_max) / 2.0
        self._dlat = self.cell_m / 110_540.0
        self._dlon = self.cell_m / max(111_320.0 * math.cos(math.radians(mid_lat)), 1.0)
        return self

    def key(self, lat, lon):
        return (int((lat - self.lat0) // self._dlat),
                int((lon - self.lon0) // self._dlon))

    def register(self, lat, lon):
        k = self.key(lat, lon)
        idx = self.cell_index.get(k)
        if idx is None:
            idx = len(self.cell_ids)
            self.cell_index[k] = idx
            self.cell_ids.append(k)
            self.cell_count.append(0)
        self.cell_count[idx] += 1
        return idx

    def lookup(self, lat, lon):
        """Índice de celda, o -1 si la celda nunca se observó."""
        return self.cell_index.get(self.key(lat, lon), -1)

    def centroid(self, idx):
        i, j = self.cell_ids[idx]
        return (self.lat0 + (i + 0.5) * self._dlat,
                self.lon0 + (j + 0.5) * self._dlon)

    def as_nodes(self):
        """Celdas como lista de dicts lat/lon, lista para urban_context."""
        out = []
        for idx in range(len(self.cell_ids)):
            lat, lon = self.centroid(idx)
            out.append({'id': idx, 'lat': lat, 'lon': lon,
                        'n_points': self.cell_count[idx]})
        return out

    @property
    def n_cells(self):
        return len(self.cell_ids)


# ─────────────────────────────────────────────────────────────────────────────
# LECTURA DEL CSV
# ─────────────────────────────────────────────────────────────────────────────

def _parse_polyline(raw):
    """POLYLINE -> lista de (lat, lon). El CSV guarda [lon, lat]."""
    if not raw or raw in ('[]', 'NA', ''):
        return []
    try:
        pts = json.loads(raw)
    except (ValueError, TypeError):
        return []
    out = []
    for p in pts:
        try:
            out.append((float(p[1]), float(p[0])))
        except (IndexError, TypeError, ValueError):
            continue
    return out


def load_trips(path,
               city_name=None,
               *,
               min_points=4,
               max_points=960,           # 4 h a 15 s
               min_dur_s=60,
               max_dur_s=4 * 3600,
               min_od_dist_m=200.0,
               max_speed_kmh=120.0,
               max_trips=None,
               keep_polyline=False,
               verbose=True):
    """
    Lee el CSV de trayectorias y devuelve la lista de viajes válidos.

    Filtros y por qué existen
    -------------------------
    MISSING_DATA=True   el propio dataset avisa de que faltan tramos GPS; la
                        etiqueta derivada del nº de puntos sería incorrecta.
    min_points          viajes de 0-3 puntos son ruido de arranque del taxímetro.
    max_points          colas de viajes anómalos (el dataset tiene trayectos de
                        varias horas que casi siempre son errores de registro).
    min_od_dist_m       origen ≈ destino: el taxi dio una vuelta o no se movió;
                        para OD TTE son casos degenerados sin señal aprovechable.
    max_speed_kmh       velocidad media en línea recta imposible -> registro malo.

    keep_polyline: guarda la geometría recorrida. NUNCA la uses como feature
    (ver el aviso de fuga en la cabecera del módulo); sirve para análisis,
    visualización o para tareas distintas de OD TTE.
    """
    path = compat.as_path(path)
    if city_name is None:
        city_name = os.path.splitext(os.path.basename(path))[0] or 'city'

    trips = []
    n_read = n_missing = n_short = n_degenerate = n_speed = 0

    with compat.open_csv(path) as fh:
        for row in csv.DictReader(fh):
            n_read += 1
            if max_trips and len(trips) >= max_trips:
                break

            if str(row.get('MISSING_DATA', 'False')).strip().lower() == 'true':
                n_missing += 1
                continue

            pts = _parse_polyline(row.get('POLYLINE'))
            n = len(pts)
            if n < min_points or n > max_points:
                n_short += 1
                continue

            dur = (n - 1) * SAMPLING_INTERVAL_S
            if not (min_dur_s <= dur <= max_dur_s):
                n_short += 1
                continue

            (o_lat, o_lon), (d_lat, d_lon) = pts[0], pts[-1]
            od = _haversine(o_lat, o_lon, d_lat, d_lon)
            if od < min_od_dist_m:
                n_degenerate += 1
                continue
            if (od / dur) * 3.6 > max_speed_kmh:
                n_speed += 1
                continue

            try:
                ts = int(row['TIMESTAMP'])
            except (KeyError, ValueError, TypeError):
                continue

            trip = {
                'trip_id': row.get('TRIP_ID', ''),
                'city': city_name,
                'taxi_id': row.get('TAXI_ID', ''),
                'call_type': (row.get('CALL_TYPE') or 'C').strip().upper(),
                'day_type': (row.get('DAY_TYPE') or 'A').strip().upper(),
                'origin_stand': _int_or_none(row.get('ORIGIN_STAND')),
                'timestamp': ts,
                'o_lat': o_lat, 'o_lon': o_lon,
                'd_lat': d_lat, 'd_lon': d_lon,
                'od_dist': od,
                'travel_time': dur,
            }
            if keep_polyline:
                trip['polyline'] = pts
            trips.append(trip)

    if verbose:
        print(f"[traj:{city_name}] {len(trips)} viajes válidos de {n_read} leídos "
              f"(descartados: {n_missing} missing, {n_short} duración, "
              f"{n_degenerate} O≈D, {n_speed} velocidad)")
    return trips


def _int_or_none(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DATASET DE UNA CIUDAD
# ─────────────────────────────────────────────────────────────────────────────

class CityTrips:
    """
    Viajes de una ciudad + su rejilla de regiones + su contexto urbano.

    Es el equivalente al antiguo 'feed' de GTFS, pero sin grafo: la estructura
    espacial ya no viene de una topología de servicio declarada sino de dónde
    circulan realmente los vehículos.
    """

    def __init__(self, trips, city_name, city_id=0, cell_m=500.0,
                 urban_ctx=None, verbose=True, corridor_mode='raster'):
        if not trips:
            raise ValueError(f"[{city_name}] no hay viajes válidos.")
        self.city = city_name
        self.city_id = city_id
        self.trips = trips
        self.urban_ctx = urban_ctx

        lats = np.array([t['o_lat'] for t in trips] + [t['d_lat'] for t in trips])
        lons = np.array([t['o_lon'] for t in trips] + [t['d_lon'] for t in trips])
        self.grid = RegionGrid(cell_m).fit(lats, lons)
        for t in trips:
            t['o_cell'] = self.grid.register(t['o_lat'], t['o_lon'])
            t['d_cell'] = self.grid.register(t['d_lat'], t['d_lon'])

        self.cells = self.grid.as_nodes()
        if urban_ctx is not None:
            urban_ctx.annotate_nodes(self.cells)
        else:
            for c in self.cells:
                c['has_poi_ctx'] = False
                c['has_road_ctx'] = False

        # extensión de la ciudad, para normalizar posiciones de forma relativa
        self.lat_span = max(self.grid.lat_max - self.grid.lat0, 1e-6)
        self.lon_span = max(self.grid.lon_max - self.grid.lon0, 1e-6)
        self.lat_c = (self.grid.lat0 + self.grid.lat_max) / 2
        self.lon_c = (self.grid.lon0 + self.grid.lon_max) / 2
        self.radius = max(_haversine(self.grid.lat0, self.grid.lon0,
                                     self.grid.lat_max, self.grid.lon_max) / 2, 1.0)

        self._annotate_corridors(mode=corridor_mode)

        if verbose:
            print(f"[traj:{city_name}] {self.grid.n_cells} celdas de "
                  f"{cell_m:.0f} m | radio urbano {self.radius/1000:.1f} km")

    # ── contexto del corredor recto O→D ──────────────────────────────────────

    def _cell_raster(self):
        """
        Agrega el contexto urbano UNA VEZ por celda, no por corredor.

        La versión exacta consulta el índice espacial cada 50 m a lo largo de
        cada corredor O-D único. Con ~500 celdas hay hasta ~250.000 pares
        posibles y eso son decenas de millones de consultas: medido, ~2 h para
        el Porto completo. La rasterización hace una sola consulta por celda
        (cientos, no millones) y después el corredor es la media ponderada de
        las celdas que atraviesa.

        Es una aproximación: pierde resolución dentro de la celda. A 500 m eso
        importa poco para densidades y proporciones, que es lo único que se
        usa, pero conviene declararlo.
        """
        ctx = self.urban_ctx
        C = self.grid.n_cells
        n_road = len(urban_context_mod.ROAD_CLASSES)
        inter = np.zeros(C)
        signal = np.zeros(C)
        poi = np.zeros(C)
        cls = np.zeros((C, n_road))
        r = self.grid.cell_m / 2.0

        for i in range(C):
            lat, lon = self.grid.centroid(i)
            if ctx.has_road:
                for k in ctx._road_idx.query(lat, lon, r):
                    rd = ctx.roads[k]
                    cls[i, urban_context_mod.ROAD_CLASSES.index(rd['cls'])] += 1
                    if rd['street_count'] >= 3:
                        inter[i] += 1
                        if rd['signal']:
                            signal[i] += 1
            if ctx.has_poi:
                poi[i] = len(ctx._poi_idx.query(lat, lon, r))

        area_km2 = (self.grid.cell_m / 1000.0) ** 2
        return {'inter': inter / area_km2, 'signal': signal / area_km2,
                'poi': poi / area_km2, 'cls': cls}

    def _cells_on_line(self, o_idx, d_idx, max_steps=200):
        """Celdas atravesadas por la recta entre dos celdas (muestreo uniforme)."""
        i0, j0 = self.grid.cell_ids[o_idx]
        i1, j1 = self.grid.cell_ids[d_idx]
        n = max(abs(i1 - i0), abs(j1 - j0))
        n = min(max(n, 1), max_steps)
        out = []
        seen = set()
        for s in range(n + 1):
            t = s / n
            key = (round(i0 + (i1 - i0) * t), round(j0 + (j1 - j0) * t))
            k = self.grid.cell_index.get(key)
            if k is not None and k not in seen:
                seen.add(k)
                out.append(k)
        return out or [o_idx, d_idx]

    def _annotate_corridors(self, mode='raster'):
        """
        Contexto urbano a lo largo de la RECTA origen-destino.

        Es deliberadamente la recta y no el trayecto recorrido: el trayecto
        recorrido codifica la respuesta (ver el aviso de fuga). La recta es
        información disponible antes del viaje, y captura lo que importa: si el
        corredor entre O y D atraviesa el centro denso o la periferia.

        mode='raster'  agrega por celda (rápido, escala al dataset completo)
        mode='exact'   consulta el índice espacial a lo largo del corredor
                       (preciso, viable solo con subconjuntos)

        Los dos modos NO son intercambiables dentro de un mismo experimento:
        'raster' trabaja en densidades por km² de celda y 'exact' en conteos por
        km de corredor, así que las columnas salen en escalas distintas. Elige
        uno y mantenlo para train, validación y test. Comparar un modelo
        entrenado en un modo contra otro entrenado en el otro no mide nada.
        """
        if self.urban_ctx is None:
            neutral = urban_context_mod.empty_edge_context()
            for t in self.trips:
                t.update(neutral)
                t['has_ctx'] = False
            return

        if mode == 'exact':
            for t in self.trips:
                poly = [(t['o_lat'], t['o_lon']), (t['d_lat'], t['d_lon'])]
                prof = self.urban_ctx.path_profile(
                    poly, cache_key=('od', t['o_cell'], t['d_cell']))
                t.update(self.urban_ctx.span_features(prof, 0, 1))
                t['has_ctx'] = True
            urban_context_mod.finalize_edge_context(self.trips)
            return

        ras = self._cell_raster()
        friction_w = np.array([urban_context_mod.ROAD_FRICTION[c]
                               for c in urban_context_mod.ROAD_CLASSES])
        cache = {}
        for t in self.trips:
            key = (t['o_cell'], t['d_cell'])
            feat = cache.get(key)
            if feat is None:
                cells = self._cells_on_line(*key)
                cl = ras['cls'][cells].sum(axis=0)
                cl_p = cl / max(cl.sum(), 1.0)
                feat = {
                    # densidades por km² -> escaladas al mismo rango que la
                    # versión exacta, que trabaja en unidades por km de corredor
                    'inter_per_km': float(ras['inter'][cells].mean()) / 20.0,
                    'signal_per_km': float(ras['signal'][cells].mean()) / 10.0,
                    'turn_per_km': 0.0,      # la recta no tiene giros
                    'circuity': 1.0,
                    'friction': float(cl_p @ friction_w),
                    '_poi_raw': float(ras['poi'][cells].mean()),
                }
                for c, v in zip(urban_context_mod.ROAD_CLASSES, cl_p):
                    feat[f'corr_{c}'] = float(v)
                cache[key] = feat
            t.update(feat)
            t['has_ctx'] = True
        urban_context_mod.finalize_edge_context(self.trips)

    # ── partición temporal ───────────────────────────────────────────────────

    def chronological_split(self, ratios=(0.8, 0.1, 0.1)):
        """
        Partición 8:1:1 por instante de salida, como TransferTraj.

        Cronológica y no aleatoria: los viajes del mismo taxi en la misma hora
        están fuertemente correlacionados, y repartirlos al azar entre train y
        test infla las métricas sin que el modelo generalice a nada.
        """
        order = sorted(range(len(self.trips)), key=lambda i: self.trips[i]['timestamp'])
        n = len(order)
        a = int(n * ratios[0])
        b = a + int(n * ratios[1])
        return order[:a], order[a:b], order[b:]

    def ood_split_by_hour(self, held_out_hours):
        """
        Escenario OOD por contexto (OOD4TTE): se excluyen del entrenamiento las
        franjas horarias indicadas y se evalúa exclusivamente en ellas.
        """
        held = set(held_out_hours)
        tr, te = [], []
        for i, t in enumerate(self.trips):
            h = int((t['timestamp'] // 3600) % 24)
            (te if h in held else tr).append(i)
        return tr, te

    def ood_split_by_distance(self, quantile=0.7):
        """
        Escenario OOD por ruta (OOD4TTE): entrenar en trayectos cortos, evaluar
        en los largos. Es la prueba más dura y la más informativa, porque el
        modelo no puede interpolar en la distancia.
        """
        d = np.array([t['od_dist'] for t in self.trips])
        cut = np.quantile(d, quantile)
        tr = [i for i in range(len(d)) if d[i] < cut]
        te = [i for i in range(len(d)) if d[i] >= cut]
        return tr, te


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR SINTÉTICO (pruebas sin descargar el dataset real)
# ─────────────────────────────────────────────────────────────────────────────

def write_synthetic_csv(out_path, city_name, n_trips=400, seed=0,
                        center=(41.15, -8.61), spread=0.05):
    """
    CSV con el mismo esquema que el de Porto. Solo para verificar el pipeline;
    los resultados obtenidos con estos datos no tienen validez experimental.
    """
    import random
    rng = random.Random(seed)
    out_path = compat.ensure_dir(out_path)
    base_ts = 1408039037

    with compat.open_csv(out_path, 'w') as f:
        w = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        w.writerow(['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND',
                    'TAXI_ID', 'TIMESTAMP', 'DAY_TYPE', 'MISSING_DATA', 'POLYLINE'])
        for k in range(n_trips):
            o_lat = center[0] + rng.gauss(0, spread / 3)
            o_lon = center[1] + rng.gauss(0, spread / 3)
            d_lat = center[0] + rng.gauss(0, spread / 3)
            d_lon = center[1] + rng.gauss(0, spread / 3)
            dist = _haversine(o_lat, o_lon, d_lat, d_lon)
            hour = rng.randint(0, 23)
            # velocidad dependiente de la hora: punta lenta, madrugada rápida
            speed = 7.0 if hour in (8, 9, 17, 18, 19) else 11.0
            speed *= rng.uniform(0.8, 1.2)
            npts = max(int(dist / speed / SAMPLING_INTERVAL_S) + 1, 5)
            if npts > 900:
                continue
            poly = []
            for s in range(npts):
                t = s / max(npts - 1, 1)
                poly.append([round(o_lon + (d_lon - o_lon) * t + rng.gauss(0, 1e-4), 6),
                             round(o_lat + (d_lat - o_lat) * t + rng.gauss(0, 1e-4), 6)])
            ts = base_ts + rng.randint(0, 60) * 86400 + hour * 3600 + rng.randint(0, 3599)
            w.writerow([f'{city_name}_T{k}', rng.choice(CALL_TYPES), 'NA',
                        str(rng.randint(1, 63)), str(20000000 + rng.randint(0, 999)),
                        str(ts), 'A', 'False', json.dumps(poly)])
    return out_path


if __name__ == '__main__':
    if len(sys.argv) > 1:
        trips = load_trips(sys.argv[1],
                           city_name=sys.argv[2] if len(sys.argv) > 2 else None)
        city = CityTrips(trips, trips[0]['city'])
        print("Ejemplo:", {k: v for k, v in city.trips[0].items()
                           if k != 'polyline'})
    else:
        p = write_synthetic_csv('/tmp/_traj_demo.csv', 'Demo', n_trips=200)
        trips = load_trips(p, city_name='Demo')
        city = CityTrips(trips, 'Demo')
        tr, va, te = city.chronological_split()
        print(f"train/val/test = {len(tr)}/{len(va)}/{len(te)}")