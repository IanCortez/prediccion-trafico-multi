"""
Carga de trayectorias de taxi con el esquema del reto ECML/PKDD 2015 (Porto) y
construcción de la rejilla de regiones de cada ciudad.

    TRIP_ID, CALL_TYPE, ORIGIN_CALL, ORIGIN_STAND, TAXI_ID, TIMESTAMP,
    DAY_TYPE, MISSING_DATA, POLYLINE      (POLYLINE = [[lon, lat], ...] cada 15 s)

Fuga de etiqueta: el tiempo de viaje es (len(POLYLINE) - 1) * 15 s, así que
cualquier feature sacada del recorrido (longitud, vértices, giros) es la
etiqueta disfrazada. Solo se usan datos conocidos antes del viaje: origen,
destino, hora de salida, tipo de llamada y el contexto de la recta O-D.
"""

import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

import compat
import urban_context as uc

SAMPLING_INTERVAL_S = 15.0

CALL_TYPES = ['A', 'B', 'C']       # A: central, B: parada, C: aleatorio
DAY_TYPES = ['A', 'B', 'C']        # A: normal, B: víspera de festivo, C: festivo


class RegionGrid:
    """Rejilla regular sobre la ciudad. Cada celda ocupada es un nodo del modelo."""

    def __init__(self, cell_m=500.0):
        self.cell_m = cell_m
        self.cell_ids = []          # (i, j) de cada celda ocupada
        self.cell_index = {}        # (i, j) -> posición en cell_ids

    def fit(self, lats, lons):
        self.lat0, self.lon0 = float(np.min(lats)), float(np.min(lons))
        self.lat_max, self.lon_max = float(np.max(lats)), float(np.max(lons))
        mid_lat = (self.lat0 + self.lat_max) / 2.0
        self._dlat = self.cell_m / 110_540.0
        self._dlon = self.cell_m / max(111_320.0 * math.cos(math.radians(mid_lat)), 1.0)
        return self

    def key(self, lat, lon):
        lat = min(max(lat, self.lat0), self.lat_max)
        lon = min(max(lon, self.lon0), self.lon_max)
        return (int((lat - self.lat0) // self._dlat),
                int((lon - self.lon0) // self._dlon))

    def register(self, lat, lon):
        k = self.key(lat, lon)
        if k not in self.cell_index:
            self.cell_index[k] = len(self.cell_ids)
            self.cell_ids.append(k)
        return self.cell_index[k]

    def centroid(self, idx):
        i, j = self.cell_ids[idx]
        return (self.lat0 + (i + 0.5) * self._dlat,
                self.lon0 + (j + 0.5) * self._dlon)

    def as_nodes(self):
        return [dict(zip(('lat', 'lon'), self.centroid(i))) for i in range(self.n_cells)]

    @property
    def n_cells(self):
        return len(self.cell_ids)


def _parse_polyline(raw):
    """POLYLINE -> [(lat, lon), ...]; el CSV la guarda como [lon, lat]."""
    if not raw or raw in ('[]', 'NA'):
        return []
    try:
        return [(float(p[1]), float(p[0])) for p in json.loads(raw)]
    except (ValueError, TypeError, IndexError):
        return []


def load_trips(path, city_name=None, *, min_dur_s=60, max_dur_s=4 * 3600,
               min_od_dist_m=200.0, max_speed_kmh=120.0, max_trips=None, verbose=True):
    """Lee el CSV y devuelve la lista de viajes válidos como dicts.

    Se descartan los viajes con MISSING_DATA, muy cortos o muy largos, con
    origen casi igual al destino, o con una velocidad en línea recta imposible.
    """
    path = compat.as_path(path)
    city_name = city_name or os.path.splitext(os.path.basename(path))[0]

    trips = []
    n_read = n_missing = n_dur = n_od = n_speed = 0
    with compat.open_csv(path) as fh:
        for row in csv.DictReader(fh):
            if max_trips and len(trips) >= max_trips:
                break
            n_read += 1
            if str(row.get('MISSING_DATA', 'False')).strip().lower() == 'true':
                n_missing += 1
                continue

            pts = _parse_polyline(row.get('POLYLINE'))
            dur = (len(pts) - 1) * SAMPLING_INTERVAL_S
            if not min_dur_s <= dur <= max_dur_s:
                n_dur += 1
                continue

            (o_lat, o_lon), (d_lat, d_lon) = pts[0], pts[-1]
            od = uc.haversine(o_lat, o_lon, d_lat, d_lon)
            if od < min_od_dist_m:
                n_od += 1
                continue
            if od / dur * 3.6 > max_speed_kmh:
                n_speed += 1
                continue
            try:
                ts = int(row['TIMESTAMP'])
            except (KeyError, ValueError, TypeError):
                continue

            trips.append({
                'call_type': (row.get('CALL_TYPE') or 'C').strip().upper(),
                'day_type': (row.get('DAY_TYPE') or 'A').strip().upper(),
                'timestamp': ts,
                'o_lat': o_lat, 'o_lon': o_lon,
                'd_lat': d_lat, 'd_lon': d_lon,
                'od_dist': od,
                'travel_time': dur,
            })

    if verbose:
        print(f"[{city_name}] {len(trips)} viajes válidos de {n_read} leídos "
              f"(descartados: {n_missing} missing, {n_dur} duración, "
              f"{n_od} O≈D, {n_speed} velocidad)")
    return trips


class CityTrips:
    """Viajes de una ciudad con su rejilla y el contexto urbano de cada celda."""

    def __init__(self, trips, city_name, cell_m=500.0, urban_ctx=None, verbose=True):
        if not trips:
            raise ValueError(f"[{city_name}] no hay viajes válidos.")
        self.city = city_name
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
                c['has_poi_ctx'] = c['has_road_ctx'] = False

        g = self.grid
        self.lat_c = (g.lat0 + g.lat_max) / 2
        self.lon_c = (g.lon0 + g.lon_max) / 2
        self.radius = max(uc.haversine(g.lat0, g.lon0, g.lat_max, g.lon_max) / 2, 1.0)

        self._annotate_corridors()
        if verbose:
            print(f"[{city_name}] {g.n_cells} celdas de {cell_m:.0f} m | "
                  f"radio {self.radius / 1000:.1f} km | "
                  f"bbox lat [{g.lat0:.4f}, {g.lat_max:.4f}] "
                  f"lon [{g.lon0:.4f}, {g.lon_max:.4f}]")

    # contexto del corredor recto O-D

    def _cell_raster(self):
        """Contexto urbano agregado una vez por celda (densidades por km²)."""
        ctx = self.urban_ctx
        C = self.grid.n_cells
        inter, signal, poi = np.zeros(C), np.zeros(C), np.zeros(C)
        cls = np.zeros((C, len(uc.ROAD_CLASSES)))
        r = self.grid.cell_m / 2.0
        for i in range(C):
            lat, lon = self.grid.centroid(i)
            if ctx.has_road:
                for k in ctx._road_idx.query(lat, lon, r):
                    rd = ctx.roads[k]
                    cls[i, uc.ROAD_CLASSES.index(rd['cls'])] += 1
                    if rd['street_count'] >= 3:
                        inter[i] += 1
                        signal[i] += bool(rd['signal'])
            if ctx.has_poi:
                poi[i] = len(ctx._poi_idx.query(lat, lon, r))
        area_km2 = (self.grid.cell_m / 1000.0) ** 2
        return {'inter': inter / area_km2, 'signal': signal / area_km2,
                'poi': poi / area_km2, 'cls': cls}

    def _cells_on_line(self, o_idx, d_idx, max_steps=200):
        """Celdas ocupadas que cruza la recta entre dos celdas."""
        i0, j0 = self.grid.cell_ids[o_idx]
        i1, j1 = self.grid.cell_ids[d_idx]
        n = min(max(abs(i1 - i0), abs(j1 - j0), 1), max_steps)
        out = []
        for s in range(n + 1):
            key = (round(i0 + (i1 - i0) * s / n), round(j0 + (j1 - j0) * s / n))
            k = self.grid.cell_index.get(key)
            if k is not None and k not in out:
                out.append(k)
        return out or [o_idx, d_idx]

    def _annotate_corridors(self):
        """Contexto urbano a lo largo de la recta O-D (no del trayecto recorrido).

        El corredor es la media de las celdas que atraviesa; se calcula una vez
        por par de celdas, no por viaje.
        """
        if self.urban_ctx is None:
            neutral = uc.empty_edge_context()
            for t in self.trips:
                t.update(neutral)
                t['has_ctx'] = False
            return

        ras = self._cell_raster()
        friction_w = np.array([uc.ROAD_FRICTION[c] for c in uc.ROAD_CLASSES])
        cache = {}
        for t in self.trips:
            key = (t['o_cell'], t['d_cell'])
            if key not in cache:
                cells = self._cells_on_line(*key)
                cl = ras['cls'][cells].sum(axis=0)
                cl_p = cl / max(cl.sum(), 1.0)
                feat = {
                    'inter_per_km': float(ras['inter'][cells].mean()) / 20.0,
                    'signal_per_km': float(ras['signal'][cells].mean()) / 10.0,
                    'friction': float(cl_p @ friction_w),
                    '_poi_raw': float(ras['poi'][cells].mean()),
                }
                for c, v in zip(uc.ROAD_CLASSES, cl_p):
                    feat[f'corr_{c}'] = float(v)
                cache[key] = feat
            t.update(cache[key])
            t['has_ctx'] = True
        uc.finalize_edge_context(self.trips)

    # grafos de regiones. Las aristas O-D salen SOLO de los viajes de
    # entrenamiento; con todos se filtraría la estructura del test.

    def _knn_pairs(self, k, max_radius_m):
        C = self.grid.n_cells
        cent = [self.grid.centroid(i) for i in range(C)]
        for i in range(C):
            near = []
            for j in range(C):
                if i != j:
                    dist = uc.haversine(*cent[i], *cent[j])
                    if dist <= max_radius_m:
                        near.append((dist, j))
            near.sort()
            for dist, j in near[:k]:
                yield i, j, dist

    def _od_flows(self, train_idx):
        flow = defaultdict(int)
        for idx in train_idx:
            t = self.trips[idx]
            if t['o_cell'] != t['d_cell']:
                flow[(t['o_cell'], t['d_cell'])] += 1
        return flow

    def build_region_graph(self, train_idx, k=8, max_radius_m=2000.0, min_flow=3):
        """Grafo simétrico ponderado: k vecinos espaciales + flujos O-D.

        Devuelve ((src, dst), peso) con la normalización D^-1/2 (A + I) D^-1/2.
        """
        w = defaultdict(float)
        for i, j, dist in self._knn_pairs(k, max_radius_m):
            ww = math.exp(-dist / self.grid.cell_m)
            w[(i, j)] = max(w[(i, j)], ww)
            w[(j, i)] = max(w[(j, i)], ww)

        flow = self._od_flows(train_idx)
        mx = max(flow.values(), default=1)
        for (i, j), c in flow.items():
            if c >= min_flow:
                ww = math.log1p(c) / math.log1p(mx)
                w[(i, j)] = max(w[(i, j)], ww)
                w[(j, i)] = max(w[(j, i)], ww)

        for i in range(self.grid.n_cells):
            w[(i, i)] = 1.0
        deg = defaultdict(float)
        for (i, _), ww in w.items():
            deg[i] += ww
        src, dst, val = [], [], []
        for (i, j), ww in w.items():
            src.append(i)
            dst.append(j)
            val.append(ww / math.sqrt(max(deg[i], 1e-8) * max(deg[j], 1e-8)))
        return (src, dst), val

    def build_directed_adjacency(self, train_idx, k=8, max_radius_m=2000.0,
                                 min_flow=3):
        """Grafo dirigido y binario para la difusión.

        Los vecinos espaciales van en ambos sentidos; los flujos O-D solo en el
        sentido del viaje, para conservar la asimetría de la demanda.
        """
        edges = set()
        for i, j, _ in self._knn_pairs(k, max_radius_m):
            edges.add((i, j))
            edges.add((j, i))
        for (i, j), c in self._od_flows(train_idx).items():
            if c >= min_flow:
                edges.add((i, j))
        return [e[0] for e in edges], [e[1] for e in edges]

    # particiones

    def chronological_split(self, ratios=(0.8, 0.1, 0.1)):
        """8:1:1 por hora de salida. Al azar, viajes casi idénticos caerían en train y test."""
        order = sorted(range(len(self.trips)), key=lambda i: self.trips[i]['timestamp'])
        a = int(len(order) * ratios[0])
        b = a + int(len(order) * ratios[1])
        return order[:a], order[a:b], order[b:]

    def ood_split_by_distance(self, quantile=0.7):
        """Entrena en los viajes cortos y evalúa en los largos."""
        d = np.array([t['od_dist'] for t in self.trips])
        cut = np.quantile(d, quantile)
        return np.flatnonzero(d < cut).tolist(), np.flatnonzero(d >= cut).tolist()
