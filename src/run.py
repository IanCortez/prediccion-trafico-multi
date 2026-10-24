"""
Entrena y evalúa el modelo sobre una ciudad objetivo, opcionalmente con
ciudades fuente.

Cada ciudad se lee de <data>/<ciudad>/train.csv. Si existen
<data>/osm/<ciudad>_pois.geojson y <data>/osm/<ciudad>_roads.geojson se usan
como contexto urbano. Las ciudades preprocesadas se guardan en cache/.
El entrenamiento corre en CUDA con semilla fija.

    python src/run.py --data src/data --city porto
    python src/run.py --data src/data --city porto --sources beijing rome
    python src/run.py --data src/data --city porto --split ood_distance \
        --st-kind tsa --seg-len 168 --corridor
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
import trajectory_loader as tl
import tte_model as M
import urban_context as uc

CORRIDOR_K = 24
CACHE_DIR = 'cache'
SEED = 0


def load_city(name, args):
    suffix = f'_n{args.max_trips}' if args.max_trips else ''
    # el nombre cambia con la rejilla recortada (BBOX_QUANTILE): las cachés viejas no se reutilizan
    cache = os.path.join(CACHE_DIR, f'{name}_c{int(args.cell_m)}_q{tl.BBOX_QUANTILE:g}{suffix}.pkl')
    if os.path.exists(cache):
        print(f"[{name}] desde caché {cache}")
        with open(cache, 'rb') as f:
            return pickle.load(f)

    csv_path = os.path.join(args.data, name, 'train.csv')
    if not os.path.exists(csv_path):
        raise SystemExit(f"No existe {csv_path}")
    trips = tl.load_trips(csv_path, name, max_trips=args.max_trips)

    pois = os.path.join(args.data, 'osm', f'{name}_pois.geojson')
    roads = os.path.join(args.data, 'osm', f'{name}_roads.geojson')
    pois = pois if os.path.exists(pois) else None
    roads = roads if os.path.exists(roads) else None
    ctx = uc.UrbanContext(pois=pois, road_nodes=roads, city=name) if (pois or roads) else None
    if ctx is None:
        print(f"[{name}] sin capas OSM: POI y red vial irán como modalidad ausente")

    city = tl.CityTrips(trips, name, cell_m=args.cell_m, urban_ctx=ctx)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache, 'wb') as f:
        pickle.dump(city, f, protocol=4)
    return city


def main():
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
    ap.add_argument('--st-kind', default='dgcn', choices=['dgcn', 'tsa'])
    ap.add_argument('--seg-len', type=int, default=6,
                    help='con tsa: 6/12/24 = Crossformer, 168 = iTransformer')
    ap.add_argument('--pace', action='store_true',
                    help='predice log d + log-ritmo (extrapola en distancia)')
    ap.add_argument('--corridor', action='store_true',
                    help='añade el ritmo del corredor O-D (implica --pace)')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--steps-per-epoch', type=int, default=None,
                    help='tope de pasos por época (por defecto, la época completa)')
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--dim', type=int, default=64)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--out', default='runs')
    args = ap.parse_args()
    args.data = compat.as_path(args.data)

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    tgt = load_city(args.city, args)
    srcs = [load_city(c, args) for c in args.sources]
    norms = M.global_norms([tgt] + srcs)
    split_kwargs = {'quantile': args.quantile} if args.split == 'ood_distance' else None
    corridor_k = CORRIDOR_K if args.corridor else None

    target = M.build_city_data(tgt, norms, args.split, split_kwargs, corridor_k=corridor_k,
                               target_fraction=args.target_fraction)
    sources = [M.build_city_data(c, norms, corridor_k=corridor_k) for c in srcs]
    print(f"train/val/test = {len(target.train_idx)}/{len(target.val_idx)}/"
          f"{len(target.test_idx)} | celdas {target.n_cells} | "
          f"cobertura ST {target.st.coverage * 100:.1f} % | fuentes {len(sources)}")

    model = M.TravelTimeModel(dim=args.dim, hidden=args.hidden,
                              bank_size=max(target.n_cells // 2, 8),
                              st_kind=args.st_kind, seg_len=args.seg_len,
                              pace=args.pace, corridor=args.corridor)
    tr = M.TSJTTrainer(model, target, sources, device='cuda', batch_size=args.batch)
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

    variant = args.st_kind + ('_corr' if args.corridor else '_pace' if args.pace else '')
    tag = f"{args.city}_{args.split}_{variant}"
    os.makedirs(args.out, exist_ok=True)
    tr.save(os.path.join(args.out, f'{tag}.pt'))
    with compat.open_text(os.path.join(args.out, f'{tag}.json'), 'w') as f:
        json.dump({**results['modelo']['total'], 'resultados': results, 'args': vars(args)},
                  f, indent=2)


def print_report(results, city, split):
    print(f"\nTEST {city} (split={split})")
    print(f"  {'':10s}{'MAE (s)':>9s}{'RMSE (s)':>10s}{'MAPE %':>8s}{'sesgo %':>9s}"
          f"{'peor hora MAE':>15s}")
    for name, r in results.items():
        t = r['total']
        print(f"  {name:10s}{t['mae']:9.1f}{t['rmse']:10.1f}{t['mape']:8.2f}{t['sesgo']:+9.1f}"
              f"{r['peor_hora']['mae']:15.1f}")
    names = list(results)
    print("\n  MAE (s) / sesgo % por tramo de distancia")
    print(f"  {'':30s}" + ''.join(f"{n:>19s}" for n in names))
    for label in results['modelo']['por_distancia']:
        row = ''.join(f"{results[n]['por_distancia'][label]['mae']:10.1f} / "
                      f"{results[n]['por_distancia'][label]['sesgo']:+6.1f}" for n in names)
        print(f"  {label:30s}{row}")


if __name__ == '__main__':
    main()
