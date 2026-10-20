"""
Descarga de OpenStreetMap las capas que usa urban_context (POIs y nodos viales)
y las guarda como GeoJSON. Se ejecuta una vez por ciudad.

    python fetch_osm.py --city porto --bbox 41.10 -8.70 41.20 -8.55 --out data/osm

La bbox va como lat_min lon_min lat_max lon_max. Requiere osmnx.

Los servidores públicos de Overpass se saturan con áreas grandes, así que:
  - los POIs se piden por teselas y con `out center` (un punto por elemento,
    sin la geometría completa de cada polígono);
  - cada tesela se guarda al terminar y los fallos se reintentan rotando de
    servidor. Si el script se corta, volver a lanzarlo retoma donde quedó.
"""

import argparse
import json
import math
import os
import time

import requests

import compat

OVERPASS_URLS = ['https://overpass-api.de/api', 'https://overpass.private.coffee/api']
OVERPASS_TIMEOUT = 300
RETRIES = 6
HEADERS = {'User-Agent': 'tesis-tte/fetch_osm (osmnx)'}

POI_KEYS = ['amenity', 'shop', 'office', 'leisure', 'tourism', 'healthcare', 'landuse']

_HIER = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary',
         'residential', 'unclassified', 'service']


def _rank(h):
    h = str(h).replace('_link', '')
    return _HIER.index(h) if h in _HIER else len(_HIER)


def _write(path, feats):
    with compat.open_text(path, 'w') as f:
        json.dump({'type': 'FeatureCollection', 'features': feats}, f)


def _retry(what, fn):
    """Llama a fn(url) rotando de servidor y esperando cada vez más tras un fallo."""
    for attempt in range(RETRIES):
        url = OVERPASS_URLS[attempt % len(OVERPASS_URLS)]
        try:
            return fn(url)
        except Exception as e:      # red, HTTP, JSON o timeout del servidor
            if attempt == RETRIES - 1:
                raise
            wait = 30 * (attempt + 1)
            print(f"  {what}: falló en {url} ({type(e).__name__}: {str(e)[:120]}); "
                  f"reintento {attempt + 1}/{RETRIES - 1} en {wait} s")
            time.sleep(wait)


def _tiles(bbox, tile_km):
    lat_min, lon_min, lat_max, lon_max = bbox
    dlat = tile_km / 110.54
    dlon = tile_km / (111.32 * math.cos(math.radians((lat_min + lat_max) / 2)))
    n_lat = max(math.ceil((lat_max - lat_min) / dlat), 1)
    n_lon = max(math.ceil((lon_max - lon_min) / dlon), 1)
    for i in range(n_lat):
        for j in range(n_lon):
            yield (i, j), (lat_min + i * dlat, lon_min + j * dlon,
                           min(lat_min + (i + 1) * dlat, lat_max),
                           min(lon_min + (j + 1) * dlon, lon_max))


def _query_pois(url, box):
    s, w, n, e = box
    sel = ''.join(f'nwr["{k}"]({s},{w},{n},{e});' for k in POI_KEYS)
    query = f'[out:json][timeout:{OVERPASS_TIMEOUT}];({sel});out tags center;'
    r = requests.post(url + '/interpreter', data={'data': query}, headers=HEADERS,
                      timeout=OVERPASS_TIMEOUT + 60)
    r.raise_for_status()
    js = r.json()
    # con timeout o falta de memoria Overpass responde 200 con un 'remark'
    if 'runtime error' in js.get('remark', ''):
        raise RuntimeError(js['remark'])
    return js['elements']


def fetch_pois(city, bbox, path, cache_dir, tile_km):
    tiles = list(_tiles(bbox, tile_km))
    print(f"[{city}] descargando POIs en {len(tiles)} teselas...")
    seen, feats = set(), []
    for (i, j), box in tiles:
        tile_path = os.path.join(cache_dir, f'{city}_pois_{i}_{j}.json')
        if os.path.exists(tile_path):
            with compat.open_text(tile_path) as f:
                elements = json.load(f)
        else:
            elements = _retry(f'tesela {i},{j}', lambda url: _query_pois(url, box))
            with compat.open_text(tile_path, 'w') as f:
                json.dump(elements, f)
        for el in elements:
            # un polígono en el borde aparece en dos teselas
            if (el['type'], el['id']) in seen:
                continue
            seen.add((el['type'], el['id']))
            pt = el if el['type'] == 'node' else el.get('center')
            props = {k: str(v) for k, v in el.get('tags', {}).items() if k in POI_KEYS}
            if pt and props:
                feats.append({'type': 'Feature',
                              'geometry': {'type': 'Point',
                                           'coordinates': [pt['lon'], pt['lat']]},
                              'properties': props})
    _write(path, feats)
    print(f"[{city}] {len(feats)} POIs")


def fetch_roads(city, bbox, path, cache_dir, tile_km, network_type='drive'):
    import osmnx as ox

    lat_min, lon_min, lat_max, lon_max = bbox
    ox.settings.cache_folder = cache_dir
    ox.settings.requests_timeout = OVERPASS_TIMEOUT
    ox.settings.max_query_area_size = (tile_km * 1000) ** 2

    def download(url):
        ox.settings.overpass_url = url
        # el endpoint /status solo es fiable en el servidor principal
        ox.settings.overpass_rate_limit = url == OVERPASS_URLS[0]
        # 'drive' deja fuera peatonales y ciclovías, que un taxi no cruza
        return ox.graph_from_bbox(bbox=(lon_min, lat_min, lon_max, lat_max),
                                  network_type=network_type)

    print(f"[{city}] descargando red vial...")
    G = _retry('red vial', download)
    feats = []
    for nid, d in G.nodes(data=True):
        # clase de la vía más importante que llega al nodo
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
    _write(path, feats)
    print(f"[{city}] {len(feats)} nodos viales")


def fetch(city, bbox, out_dir, network_type='drive', tile_km=10.0):
    out_dir = compat.as_path(out_dir)
    cache_dir = os.path.join(out_dir, '_overpass_cache')
    os.makedirs(cache_dir, exist_ok=True)

    roads_path = os.path.join(out_dir, f'{city}_roads.geojson')
    pois_path = os.path.join(out_dir, f'{city}_pois.geojson')
    if os.path.exists(roads_path):
        print(f"[{city}] {roads_path} ya existe, se omite")
    else:
        fetch_roads(city, bbox, roads_path, cache_dir, tile_km, network_type)
    if os.path.exists(pois_path):
        print(f"[{city}] {pois_path} ya existe, se omite")
    else:
        fetch_pois(city, bbox, pois_path, cache_dir, tile_km)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--city', required=True)
    ap.add_argument('--bbox', nargs=4, type=float, required=True,
                    metavar=('LAT_MIN', 'LON_MIN', 'LAT_MAX', 'LON_MAX'))
    ap.add_argument('--out', default='data/osm')
    ap.add_argument('--network', default='drive')
    ap.add_argument('--tile-km', type=float, default=10.0,
                    help='lado de cada tesela consultada a Overpass')
    a = ap.parse_args()
    fetch(a.city, a.bbox, a.out, a.network, a.tile_km)
