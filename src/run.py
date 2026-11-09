"""
Entrena y evalúa el modelo sobre una ciudad objetivo, opcionalmente con
ciudades fuente.

Cada ciudad se lee de <data>/<ciudad>/train.csv. Si existen
<data>/osm/<ciudad>_pois.geojson y <data>/osm/<ciudad>_roads.geojson se usan
como contexto urbano. Las ciudades preprocesadas se guardan en cache/.
El entrenamiento corre en CUDA con semilla fija.

    python src/run.py --data src/data --city porto --cell-m 1500
    python src/run.py --data src/data --city porto --cell-m 1500 --sources beijing_v2 rome
    python src/run.py --data src/data --city porto --cell-m 1500 --split ood_distance --pace
    python src/run.py --data src/data --city porto --cell-m 1500 --sota-filter --posenc

--sota-filter deja en la objetivo solo los viajes de la zona central de Porto con
5 a 120 puntos, la definición de viaje de las cifras publicadas.
--posenc agrega la posición fina de O y D (ver tte_model.POSENC_NAMES).
"""

import argparse
import json
import os
import pickle
import random

import numpy as np
import torch

import baselines as B
import compat
import road_network as RN
import trajectory_loader as tl
import tte_model as M
import urban_context as uc

CACHE_DIR = 'cache'
# v2: hora local por ciudad, longitud real del recorrido (objetivo auxiliar) y red vial
CACHE_VERSION = 'v2'
SEED = 0

# Definición de viaje de las cifras publicadas para Porto (ver
# docs/transferencia_ood.md): O y D dentro de la zona central de
# 11,13 x 5,11 km y trayectorias de 5 a 120 puntos. El mínimo de 5 puntos (60 s)
# ya lo aplica load_trips; aquí se agrega el máximo de 120 (119 * 15 s).
SOTA_FILTER = {
    'porto': {'lat': (41.1405, 41.1859), 'lon': (-8.6887, -8.5557), 'max_s': 119 * 15},
}


def sota_filter(trips, name):
    if name not in SOTA_FILTER:
        raise SystemExit(f"--sota-filter no está definido para {name!r}")
    f = SOTA_FILTER[name]
    inside = lambda lat, lon: (f['lat'][0] <= lat <= f['lat'][1]
                               and f['lon'][0] <= lon <= f['lon'][1])
    kept = [t for t in trips if t['travel_time'] <= f['max_s']
            and inside(t['o_lat'], t['o_lon']) and inside(t['d_lat'], t['d_lon'])]
    print(f"[{name}] filtro SOTA: {len(kept)} de {len(trips)} viajes")
    return kept


def use_posenc(args):
    # las corridas guardadas antes de --posenc lo registran como extra_feats=['posenc']
    return getattr(args, 'posenc', False) or 'posenc' in (getattr(args, 'extra_feats', None) or [])


def load_city(name, args):
    osm = os.path.join(args.data, 'osm')
    layers = [os.path.join(osm, f'{name}_{k}') for k in ('pois.geojson', 'roads.geojson', 'edges.npz')]
    pois, roads, _ = [p if os.path.exists(p) else None for p in layers]
    # el nombre cambia con la rejilla (BBOX_QUANTILE), la versión de CityTrips y
    # las capas OSM presentes (p, r, e), para no reutilizar cachés construidas
    # con otra lógica o antes de descargar OSM
    present = ''.join(k for k, p in zip('pre', layers) if os.path.exists(p))
    # el filtro SOTA se aplica solo a la ciudad objetivo; las fuentes quedan completas
    sota = getattr(args, 'sota_filter', False) and name == args.city
    suffix = ((f'_osm{present}' if present else '') + (f'_n{args.max_trips}' if args.max_trips else '')
              + ('_sota' if sota else ''))
    cache = os.path.join(CACHE_DIR, f'{name}_c{int(args.cell_m)}_q{tl.BBOX_QUANTILE:g}'
                                    f'_{CACHE_VERSION}{suffix}.pkl')
    if os.path.exists(cache):
        print(f"[{name}] desde caché {cache}")
        with open(cache, 'rb') as f:
            return pickle.load(f)

    csv_path = os.path.join(args.data, name, 'train.csv')
    if not os.path.exists(csv_path):
        raise SystemExit(f"No existe {csv_path}")
    trips = tl.load_trips(csv_path, name, max_trips=args.max_trips)
    if sota:
        # se filtra después de leer: con --max-trips quedan menos de max_trips viajes
        trips = sota_filter(trips, name)

    ctx = uc.UrbanContext(pois=pois, road_nodes=roads, city=name) if (pois or roads) else None
    if ctx is None:
        print(f"[{name}] sin capas OSM: POI y red vial irán como modalidad ausente")

    road = RN.load_graph(osm, name)
    if road is None:
        print(f"[{name}] sin {name}_edges.npz: la distancia por red vial irá como ausente")
    city = tl.CityTrips(trips, name, cell_m=args.cell_m, urban_ctx=ctx, road_graph=road)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache, 'wb') as f:
        pickle.dump(city, f, protocol=4)
    return city


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='carpeta con <ciudad>/train.csv')
    ap.add_argument('--city', required=True, help='ciudad objetivo')
    ap.add_argument('--sources', nargs='*', default=[], help='ciudades fuente')
    ap.add_argument('--cell-m', type=float, default=500.0)
    ap.add_argument('--max-trips', type=int, default=None)
    ap.add_argument('--split', default='chronological',
                    choices=['chronological', 'ood_distance'])
    ap.add_argument('--quantile', type=float, default=0.7,
                    help='corte de distancia con --split ood_distance')
    ap.add_argument('--target-fraction', type=float, default=None,
                    help='fracción del train objetivo que se usa (few-shot)')
    ap.add_argument('--pace', action='store_true',
                    help='predice log d + log-ritmo (extrapola en distancia)')
    ap.add_argument('--sota-filter', action='store_true',
                    help='solo viajes de la zona central y de 5 a 120 puntos (solo porto)')
    ap.add_argument('--posenc', action='store_true',
                    help='posición fina de O y D: senos y cosenos multiescala en km')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--steps-per-epoch', type=int, default=None,
                    help='tope de pasos por época (por defecto, la época completa)')
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--source-ratio', type=float, default=1.0,
                    help='peso del gradiente de las fuentes respecto del de la objetivo')
    ap.add_argument('--lambda-align', type=float, default=0.1,
                    help='peso de la alineación MMD de embeddings de celda entre ciudades')
    ap.add_argument('--lambda-aux', type=float, default=0.1,
                    help='peso del objetivo auxiliar de circuidad del recorrido (solo entrenamiento)')
    ap.add_argument('--dim', type=int, default=64)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--out', default='runs')
    return ap


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def prepare(args):
    """Ciudades (CityTrips) y sus tensores (CityData) de la objetivo y de las fuentes."""
    args.data = compat.as_path(args.data)
    tgt = load_city(args.city, args)
    srcs = [load_city(c, args) for c in args.sources]
    split_kwargs = {'quantile': args.quantile} if args.split == 'ood_distance' else None
    # la normalización sale del train de la objetivo (ya recortado si es few-shot) y las
    # fuentes usan la misma: así la objetivo se ve igual con o sin fuentes, y ni el
    # test ni los viajes descartados en few-shot entran en ella
    posenc = use_posenc(args)
    target = M.build_city_data(tgt, None, args.split, split_kwargs,
                               target_fraction=args.target_fraction, posenc=posenc)
    sources = [M.build_city_data(c, target.norms, posenc=posenc) for c in srcs]
    print(f"train/val/test = {len(target.train_idx)}/{len(target.val_idx)}/"
          f"{len(target.test_idx)} | celdas {target.n_cells} | "
          f"cobertura ST {target.st.coverage * 100:.1f} % | fuentes {len(sources)}")
    return tgt, srcs, target, sources


def build_trainer(args, target, sources):
    n_extra = len(M.POSENC_NAMES) if use_posenc(args) else 0
    model = M.TravelTimeModel(dim=args.dim, hidden=args.hidden,
                              bank_size=max(target.n_cells // 2, 8), pace=args.pace,
                              n_extra=n_extra)
    # getattr: las corridas guardadas antes de existir el argumento usan el valor original
    return M.Trainer(model, target, sources, device='cuda', batch_size=args.batch,
                         source_ratio=getattr(args, 'source_ratio', 0.005),
                         lambda_align=getattr(args, 'lambda_align', 0.0),
                         lambda_aux=getattr(args, 'lambda_aux', 0.0))


def load_run(json_path):
    """Reconstruye una corrida guardada (datos + modelo entrenado) a partir de su JSON.

    Devuelve (args, tgt, target, sources, trainer); el .pt debe estar junto al JSON.
    """
    with compat.open_text(json_path) as f:
        saved = json.load(f)
    args = argparse.Namespace(**saved['args'])
    tgt, srcs, target, sources = prepare(args)
    tr = build_trainer(args, target, sources)
    ckpt = torch.load(os.path.splitext(json_path)[0] + '.pt', map_location='cuda')
    tr.model.load_state_dict(ckpt['model_state'])
    tr.history = ckpt['history']
    return args, tgt, target, sources, tr


def run(args, seed=SEED):
    """Entrena, evalúa contra los baselines y guarda los resultados en args.out.

    Devuelve las métricas de test del modelo y de los baselines, y las épocas entrenadas.
    """
    set_seed(seed)
    tgt, srcs, target, sources = prepare(args)
    tr = build_trainer(args, target, sources)
    tr.train(epochs=args.epochs, patience=args.patience, steps_per_epoch=args.steps_per_epoch)

    # el modelo y los baselines se evalúan sobre el mismo test y con el mismo train
    test = np.asarray(target.test_idx)
    true = target.raw_times[test].numpy()
    hour = target.st.how.cpu().numpy()[test] % 24
    dist = target.od_dist[test].numpy()
    preds = {'modelo': tr.predict(target, target.test_idx)}
    preds.update(B.predict(tgt, target.train_idx, target.test_idx))
    results = {name: M.report(p, true, hour, dist) for name, p in preds.items()}
    print_report(results, target.city, args.split)

    # 'dgcn' se conserva en el nombre para que las corridas anteriores sigan encontrándose
    tag = f"{args.city}_{args.split}_dgcn" + ('_pace' if args.pace else '')
    os.makedirs(args.out, exist_ok=True)
    tr.save(os.path.join(args.out, f'{tag}.pt'))
    with compat.open_text(os.path.join(args.out, f'{tag}.json'), 'w') as f:
        json.dump({**results['modelo']['total'], 'resultados': results,
                   'semilla': seed, 'epocas': len(tr.history['val_mae']),
                   'viajes_train': len(target.train_idx), 'args': vars(args)},
                  f, indent=2)
    return results, len(tr.history['val_mae'])


def main():
    run(build_parser().parse_args())


def print_report(results, city, split):
    print(f"\nTEST {city} (split={split})")
    print(f"  {'':10s}{'MAE (s)':>9s}{'RMSE (s)':>10s}{'MdAE (s)':>10s}{'MAPE %':>8s}"
          f"{'sesgo %':>9s}{'peor hora MAE':>15s}")
    for name, r in results.items():
        t = r['total']
        print(f"  {name:10s}{t['mae']:9.1f}{t['rmse']:10.1f}{t['mdae']:10.1f}{t['mape']:8.2f}"
              f"{t['sesgo']:+9.1f}{r['peor_hora']['mae']:15.1f}")
    # entre datasets o splits distintos los segundos no se comparan; la mejora relativa sí
    mae = results['modelo']['total']['mae']
    # los trabajos publicados sobre Porto reportan MAE y RMSE en minutos
    print(f"  MAE del modelo en minutos: {mae / 60:.3f} | "
          f"RMSE {results['modelo']['total']['rmse'] / 60:.3f}")
    print("  MAE del modelo frente a: " + " | ".join(
        f"{n} {(mae / r['total']['mae'] - 1) * 100:+.1f} %" for n, r in results.items() if n != 'modelo'))
    names = list(results)
    print("\n  MAE (s) / sesgo % por franja fija de distancia")
    print(f"  {'':22s}{'n':>8s}" + ''.join(f"{n:>19s}" for n in names))
    for label, m in results['modelo']['por_franja_km'].items():
        if not m['n']:
            continue
        row = ''.join(f"{results[n]['por_franja_km'][label]['mae']:10.1f} / "
                      f"{results[n]['por_franja_km'][label]['sesgo']:+6.1f}" for n in names)
        print(f"  {label:22s}{m['n']:8d}{row}")


if __name__ == '__main__':
    main()
