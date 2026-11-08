"""
Red vial de OSM y medidas por par de celdas: circuidad de la red y ritmo a
velocidad libre.

Las aristas de cada ciudad se guardan en <osm>/<ciudad>_edges.npz. fetch_osm.py
las escribe al descargar la red vial; para una ciudad ya descargada se pueden
reconstruir desde la caché de Overpass, sin conexión:

    python src/road_network.py --city porto --osm src/data/osm

Al modelo solo entran medidas relativas (desvío de la red respecto de la recta
y segundos por metro a velocidad libre). No escalan con la distancia, así que
se transfieren entre ciudades y no rompen la extrapolación de --pace.
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

import compat

# velocidad libre por clase de vía (km/h) cuando la vía no trae maxspeed; las
# vías fuera de esta tabla (peatonales, ciclovías...) no se usan
FREE_FLOW_KMH = {
    'motorway': 100, 'motorway_link': 60, 'trunk': 80, 'trunk_link': 50,
    'primary': 60, 'primary_link': 40, 'secondary': 50, 'secondary_link': 35,
    'tertiary': 40, 'tertiary_link': 30, 'unclassified': 30, 'residential': 30,
    'living_street': 10, 'service': 20, 'road': 30,
}
SNAP_KMH = 20.0          # para el tramo entre el centro de la celda y el nodo más cercano
MAX_SNAP_M = 1000.0      # más lejos de cualquier vía, la celda queda sin red
PACE_REF = np.log(3.6 / 50)   # log(s/m) a 50 km/h: centra el ritmo libre cerca de 0


def _speed(highway, maxspeed):
    try:
        return float(str(maxspeed).split(';')[0].split()[0])
    except (TypeError, ValueError, IndexError):
        return FREE_FLOW_KMH.get(highway)


def _haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 6_371_000.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class RoadGraph:
    """Grafo no dirigido de vías para auto: nodos (lat, lon) y aristas con longitud y velocidad."""

    def __init__(self, lat, lon, u, v, length, speed_kmh):
        self.lat, self.lon = np.asarray(lat, float), np.asarray(lon, float)
        self.u, self.v = np.asarray(u, np.int64), np.asarray(v, np.int64)
        self.length = np.asarray(length, float)
        self.speed = np.asarray(speed_kmh, float)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z['lat'], z['lon'], z['u'], z['v'], z['length'], z['speed'])

    def save(self, path):
        np.savez_compressed(path, lat=self.lat, lon=self.lon, u=self.u, v=self.v,
                            length=self.length, speed=self.speed)

    @classmethod
    def from_overpass(cls, elements, bbox=None):
        """Desde elementos JSON de Overpass (nodos y vías con etiqueta highway)."""
        coord = {e['id']: (e['lat'], e['lon']) for e in elements
                 if e.get('type') == 'node' and 'lat' in e}
        if bbox is not None:
            lat0, lon0, lat1, lon1 = bbox
            coord = {k: c for k, c in coord.items() if lat0 <= c[0] <= lat1 and lon0 <= c[1] <= lon1}
        edges = {}
        for e in elements:
            tags = e.get('tags', {})
            if e.get('type') != 'way' or tags.get('highway') not in FREE_FLOW_KMH:
                continue
            speed = _speed(tags.get('highway'), tags.get('maxspeed'))
            nodes = [n for n in e.get('nodes', []) if n in coord]
            for a, b in zip(nodes[:-1], nodes[1:]):
                if a != b:
                    edges[(min(a, b), max(a, b))] = speed
        return cls._from_edges(coord, edges)

    @classmethod
    def from_osmnx(cls, G):
        coord = {n: (d['y'], d['x']) for n, d in G.nodes(data=True)}
        edges = {}
        for a, b, d in G.edges(data=True):
            hw = d.get('highway')
            hw = hw[0] if isinstance(hw, list) else hw
            if hw in FREE_FLOW_KMH and a != b:
                edges[(min(a, b), max(a, b))] = _speed(hw, d.get('maxspeed'))
        return cls._from_edges(coord, edges)

    @classmethod
    def _from_edges(cls, coord, edges):
        used = sorted({n for e in edges for n in e})
        index = {n: i for i, n in enumerate(used)}
        lat = np.array([coord[n][0] for n in used])
        lon = np.array([coord[n][1] for n in used])
        u = np.array([index[a] for a, _ in edges])
        v = np.array([index[b] for _, b in edges])
        length = _haversine(lat[u], lon[u], lat[v], lon[v])
        return cls(lat, lon, u, v, length, np.array(list(edges.values()), float))

    def cell_matrices(self, cell_lat, cell_lon, chunk=256):
        """Distancia (m) y tiempo a velocidad libre (s) por la red entre cada par de puntos.

        Cada punto se une al nodo más cercano; el tramo hasta él se suma a
        SNAP_KMH. Pares sin camino quedan en inf.
        """
        n = len(self.lat)
        g_len = csr_matrix((self.length, (self.u, self.v)), shape=(n, n))
        g_time = csr_matrix((self.length / (self.speed / 3.6), (self.u, self.v)), shape=(n, n))
        # solo el componente conexo más grande: las teselas de Overpass dejan
        # fragmentos sueltos y unir un punto a uno de ellos daría pares sin camino
        _, comp = connected_components(g_len, directed=False)
        main = np.flatnonzero(comp == np.bincount(comp).argmax())
        k = np.cos(np.radians(np.mean(self.lat)))
        tree = cKDTree(np.c_[self.lat[main], self.lon[main] * k])
        _, nearest = tree.query(np.c_[cell_lat, cell_lon * k])
        snap = main[nearest]
        snap_m = _haversine(cell_lat, cell_lon, self.lat[snap], self.lon[snap])
        src, inv = np.unique(snap, return_inverse=True)
        D = np.empty((len(cell_lat), len(cell_lat)), np.float32)
        T = np.empty_like(D)
        for start in range(0, len(src), chunk):
            rows = src[start:start + chunk]
            d = dijkstra(g_len, directed=False, indices=rows)[:, snap]
            t = dijkstra(g_time, directed=False, indices=rows)[:, snap]
            for r, node in enumerate(rows):
                cells = np.flatnonzero(snap == node)
                D[cells] = d[r] + snap_m[cells, None] + snap_m[None]
                T[cells] = t[r] + (snap_m[cells, None] + snap_m[None]) / (SNAP_KMH / 3.6)
        # un punto a más de MAX_SNAP_M de cualquier vía está fuera de la red descargada
        far = snap_m > MAX_SNAP_M
        D[far, :] = D[:, far] = np.inf
        T[far, :] = T[:, far] = np.inf
        return D, T


def load_graph(osm_dir, city):
    path = os.path.join(osm_dir, f'{city}_edges.npz')
    return RoadGraph.load(path) if os.path.exists(path) else None


def _bbox_from_roads(osm_dir, city):
    with compat.open_text(os.path.join(osm_dir, f'{city}_roads.geojson')) as f:
        pts = np.array([ft['geometry']['coordinates'] for ft in json.load(f)['features']])
    return pts[:, 1].min(), pts[:, 0].min(), pts[:, 1].max(), pts[:, 0].max()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--city', required=True)
    ap.add_argument('--osm', default='src/data/osm')
    args = ap.parse_args()
    bbox = _bbox_from_roads(args.osm, args.city)
    elements = []
    for path in glob.glob(os.path.join(args.osm, '_overpass_cache', '*.json')):
        with compat.open_text(path) as f:
            d = json.load(f)
        elements += d.get('elements', []) if isinstance(d, dict) else []
    g = RoadGraph.from_overpass(elements, bbox)
    out = os.path.join(args.osm, f'{args.city}_edges.npz')
    g.save(out)
    print(f"[{args.city}] {len(g.lat)} nodos, {len(g.u)} aristas, "
          f"{g.length.sum() / 1000:.0f} km de vías -> {out}")


if __name__ == '__main__':
    main()
