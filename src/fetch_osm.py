"""
fetch_osm.py
============
Extrae de OpenStreetMap las dos capas que consume urban_context y las vuelca a
GeoJSON. Se ejecuta UNA VEZ por ciudad; el entrenamiento nunca toca la red.

    python fetch_osm.py --city porto --bbox 41.10 -8.70 41.20 -8.55 --out data/osm

Produce:
    data/osm/porto_pois.geojson    puntos de interés con sus etiquetas OSM
    data/osm/porto_roads.geojson   nodos viales con street_count y semáforos

Requiere: pip install osmnx
La bbox se da como  lat_min lon_min lat_max lon_max.

Sobre el tamaño: para un área urbana de ~15×15 km salen del orden de decenas de
miles de POIs y de nodos viales. Ambos archivos caben de sobra en memoria y
urban_context los indexa por rejilla en segundos.
"""

import argparse
import json
import os

import compat


POI_TAGS = {
    'amenity': True, 'shop': True, 'office': True, 'leisure': True,
    'tourism': True, 'healthcare': True, 'landuse': True,
}


def fetch(city, bbox, out_dir, network_type='drive'):
    import osmnx as ox

    lat_min, lon_min, lat_max, lon_max = bbox
    out_dir = compat.as_path(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ── red vial ─────────────────────────────────────────────────────────────
    # network_type='drive' excluye peatonales y ciclovías: para tiempos de viaje
    # en vehículo, incluirlas infla la densidad de intersecciones con cruces por
    # los que el taxi nunca pasa.
    print(f"[{city}] descargando red vial…")
    G = ox.graph_from_bbox(bbox=(lon_min, lat_min, lon_max, lat_max),
                           network_type=network_type)
    G = ox.add_edge_speeds(G) if hasattr(ox, 'add_edge_speeds') else G

    feats = []
    for nid, d in G.nodes(data=True):
        # jerarquía de la vía más importante que incide en el nodo: determina
        # la clase canónica de la celda en urban_context
        hw = None
        for _, _, e in G.edges(nid, data=True):
            h = e.get('highway')
            if isinstance(h, list):
                h = h[0]
            if h and (hw is None or _rank(h) < _rank(hw)):
                hw = h
        feats.append({
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [d['x'], d['y']]},
            'properties': {
                'street_count': int(d.get('street_count', 3)),
                'signal': str(d.get('highway', '')).lower() in (
                    'traffic_signals', 'crossing', 'stop', 'mini_roundabout'),
                'highway': hw or 'residential',
            },
        })
    _write(os.path.join(out_dir, f'{city}_roads.geojson'), feats)
    print(f"[{city}] {len(feats)} nodos viales")

    # ── POIs ─────────────────────────────────────────────────────────────────
    print(f"[{city}] descargando POIs…")
    gdf = ox.features_from_bbox(bbox=(lon_min, lat_min, lon_max, lat_max),
                                tags=POI_TAGS)
    feats = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        # los polígonos (edificios, parques) entran por su centroide
        pt = geom if geom.geom_type == 'Point' else geom.centroid
        props = {k: str(row[k]) for k in POI_TAGS
                 if k in row and row[k] is not None and str(row[k]) != 'nan'}
        if not props:
            continue
        feats.append({
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [pt.x, pt.y]},
            'properties': props,
        })
    _write(os.path.join(out_dir, f'{city}_pois.geojson'), feats)
    print(f"[{city}] {len(feats)} POIs")


_HIER = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary',
         'residential', 'unclassified', 'service']


def _rank(h):
    h = str(h).replace('_link', '')
    return _HIER.index(h) if h in _HIER else len(_HIER)


def _write(path, feats):
    with compat.open_text(path, 'w') as f:
        json.dump({'type': 'FeatureCollection', 'features': feats}, f)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--city', required=True)
    ap.add_argument('--bbox', nargs=4, type=float, required=True,
                    metavar=('LAT_MIN', 'LON_MIN', 'LAT_MAX', 'LON_MAX'))
    ap.add_argument('--out', default='data/osm')
    ap.add_argument('--network', default='drive')
    a = ap.parse_args()
    fetch(a.city, a.bbox, a.out, a.network)