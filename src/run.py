"""
run_experiment.py
=================
Ejecuta el pipeline completo por etapas. Cada etapa cachea su salida, de modo
que se puede repetir el entrenamiento sin volver a preprocesar 1,7 M de viajes.

    # 0. comprobar que todo funciona con una muestra pequeña
    python run_experiment.py preflight --data data/porto/train.csv

    # 1. preprocesar y cachear (una vez)
    python run_experiment.py prepare --data data/porto/train.csv --city porto \
           --pois data/osm/porto_pois.geojson --roads data/osm/porto_roads.geojson

    # 2. entrenar y evaluar
    python run_experiment.py train --city porto --split chronological
    python run_experiment.py train --city porto --split ood_distance
    python run_experiment.py train --city porto --split ood_hour --hours 7 8 9

    # 3. ablación de componentes
    python run_experiment.py ablate --city porto
"""

import argparse
import json
import os
import pickle
import random
import time

import numpy as np
import torch

import compat
import trajectory_loader as tl
import urban_context as uc
import tte_model as M


CACHE_DIR = 'cache'


def _cache_path(city, cell_m, corridor_mode):
    return os.path.join(CACHE_DIR, f'{city}_c{int(cell_m)}_{corridor_mode}.pkl')


def _norm_args(a):
    """Expande '~' y normaliza separadores en todas las rutas de la CLI."""
    for k in ('data', 'pois', 'roads'):
        v = getattr(a, k, None)
        if v:
            setattr(a, k, compat.as_path(v))
    return a


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 0 — PREFLIGHT
# ─────────────────────────────────────────────────────────────────────────────

def preflight(args):
    """
    Lee solo las primeras N filas y comprueba lo que suele romperse: columnas
    con otro nombre, POLYLINE mal escapada, timestamps en milisegundos, tasa de
    descarte absurda. Cuesta segundos y evita descubrir el problema dos horas
    después de lanzar el preprocesado completo.
    """
    t0 = time.time()
    trips = tl.load_trips(args.data, city_name='preflight',
                          max_trips=args.n, verbose=True)
    if not trips:
        raise SystemExit("Ningún viaje válido. Revisa el separador, las "
                         "comillas de POLYLINE y el nombre de las columnas.")

    ts = [t['timestamp'] for t in trips]
    if min(ts) > 3e10:
        print("AVISO: los TIMESTAMP parecen milisegundos, no segundos. "
              "La hora del día saldrá mal — divide entre 1000 antes.")
    dur = np.array([t['travel_time'] for t in trips])
    dist = np.array([t['od_dist'] for t in trips])
    spd = dist / dur * 3.6
    print(f"\n  duración   mediana {np.median(dur):6.0f} s   p95 {np.percentile(dur,95):6.0f} s")
    print(f"  dist O-D   mediana {np.median(dist):6.0f} m   p95 {np.percentile(dist,95):6.0f} m")
    print(f"  vel media  mediana {np.median(spd):6.1f} km/h")
    print(f"  rango de fechas: {time.strftime('%Y-%m-%d', time.gmtime(min(ts)))}"
          f" .. {time.strftime('%Y-%m-%d', time.gmtime(max(ts)))}")

    city = tl.CityTrips(trips, 'preflight', verbose=True)
    print(f"\n  bbox lat [{city.grid.lat0:.4f}, {city.grid.lat_max:.4f}]  "
          f"lon [{city.grid.lon0:.4f}, {city.grid.lon_max:.4f}]")
    print("  ^ usa esta bbox en fetch_osm.py")
    print(f"\nPreflight OK en {time.time()-t0:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 1 — PREPARE
# ─────────────────────────────────────────────────────────────────────────────

def prepare(args):
    os.makedirs(CACHE_DIR, exist_ok=True)
    t0 = time.time()

    trips = tl.load_trips(args.data, city_name=args.city,
                          max_trips=args.max_trips, verbose=True)
    print(f"  lectura: {time.time()-t0:.1f}s")

    ctx = None
    if args.pois or args.roads:
        t1 = time.time()
        # el contexto se construye SIN la ciudad todavía: solo necesita las
        # capas OSM, y anotará las celdas cuando CityTrips se las pase
        ctx = uc.UrbanContext(pois=args.pois, road_nodes=args.roads,
                              city=args.city)
        print(f"  capas OSM indexadas: {time.time()-t1:.1f}s")
    else:
        print("  sin capas OSM: las modalidades POI y vial irán enmascaradas")

    t2 = time.time()
    city = tl.CityTrips(trips, args.city, cell_m=args.cell_m, urban_ctx=ctx,
                        corridor_mode=args.corridor_mode, verbose=True)
    print(f"  celdas + contexto: {time.time()-t2:.1f}s")

    # se cachea el objeto ya anotado; la polilínea no está (keep_polyline=False)
    # así que el pickle es manejable incluso con el dataset completo
    path = _cache_path(args.city, args.cell_m, args.corridor_mode)
    with open(path, 'wb') as f:
        pickle.dump(city, f, protocol=4)
    print(f"\nCacheado → {path}  ({os.path.getsize(path)/1e6:.0f} MB)  "
          f"total {time.time()-t0:.1f}s")


def _load_city(city_name, cell_m, corridor_mode):
    path = _cache_path(city_name, cell_m, corridor_mode)
    if not os.path.exists(path):
        raise SystemExit(f"Falta {path}. Ejecuta primero la etapa 'prepare'.")
    with open(path, 'rb') as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 2 — TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def _build(args):
    tgt = _load_city(args.city, args.cell_m, args.corridor_mode)
    srcs = [_load_city(c, args.cell_m, args.corridor_mode) for c in args.sources]
    dm, ds, tm, ts = M.global_norms([tgt] + srcs)

    kw = {}
    if args.split == 'ood_hour':
        kw = {'held_out_hours': args.hours}
    elif args.split == 'ood_distance':
        kw = {'quantile': args.quantile}

    kw_st = dict(use_graph=not args.no_graph, graph_k=args.graph_k,
                 use_stgnn=not args.no_stgnn)
    target = M.build_city_data(tgt, 0, dm, ds, tm, ts, args.split, kw, **kw_st)
    sources = [M.build_city_data(c, i + 1, dm, ds, tm, ts, 'chronological', **kw_st)
               for i, c in enumerate(srcs)]
    if target.st is not None:
        print(f"tensor espaciotemporal: 168 h x {target.n_cells} celdas | "
              f"cobertura {target.st['coverage']*100:.1f} % de celdas-hora con datos")
    if args.target_fraction:
        k = max(int(len(target.train_idx) * args.target_fraction), 8)
        target.train_idx = target.train_idx[:k]
    return target, sources


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    target, sources = _build(args)
    n_e = 0 if target.edge_weight is None else len(target.edge_weight)
    print(f"train/val/test = {len(target.train_idx)}/{len(target.val_idx)}/"
          f"{len(target.test_idx)}  | celdas {target.n_cells}"
          f"  | aristas {n_e}  | fuentes {len(sources)}")

    model = M.TravelTimeModel(dim=args.dim, hidden=args.hidden,
                              bank_size=max(target.n_cells // 2, 8),
                              invariant=not args.no_invariant,
                              use_npm=not args.no_npm,
                              n_graph_layers=0 if args.no_graph
                              else args.graph_layers,
                              use_stgnn=not args.no_stgnn,
                              st_window=args.st_window, st_K=args.st_K)
    tr = M.TSJTTrainer(model, target, sources,
                       device=args.device if args.device != 'auto'
                       else ('cuda' if torch.cuda.is_available() else 'cpu'),
                       alpha=args.alpha, beta=args.beta,
                       batch_size=args.batch,
                       target_per_batch=args.target_per_batch,
                       use_tsb=not args.no_tsb,
                       lambda_st=args.lambda_st, st_mask_ratio=args.st_mask)
    tr.train(epochs=args.epochs, patience=args.patience,
             steps_per_epoch=args.steps_per_epoch)

    mae, rmse, mape = tr.evaluate(target, target.test_idx)
    print(f"\n── TEST {target.city} · split={args.split} ──")
    print(f"  MAE {mae:8.1f} s | RMSE {rmse:8.1f} s | MAPE {mape:6.2f} %")
    print("\n── por franja horaria ──")
    for k, v in tr.evaluate_by_hour(target, target.test_idx).items():
        print(f"  {k:<12} MAE {v[0]:7.1f} s | MAPE {v[2]:6.2f} %")

    os.makedirs('runs', exist_ok=True)
    tag = f"{args.city}_{args.split}_s{args.seed}"
    tr.save(f'runs/{tag}.pt')
    with compat.open_text(f'runs/{tag}.json', 'w') as f:
        json.dump({'mae': mae, 'rmse': rmse, 'mape': mape,
                   'args': vars(args)}, f, indent=2, default=str)
    return tr, target


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 3 — ABLACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def ablate(args):
    """
    Ablación completa. Cada variante se corre con las mismas semillas; si no,
    la diferencia medida puede ser ruido de inicialización en lugar del efecto
    del componente.
    """
    variants = [
        ('completo', {}),
        ('wo TSB', {'no_tsb': True}),
        ('wo grafo', {'no_graph': True}),
        ('wo ST-GNN', {'no_stgnn': True}),
        ('wo interpolacion', {'lambda_st': 0.0}),
        ('wo NPM', {'no_npm': True}),
        ('wo invariancia', {'no_invariant': True}),
        ('wo contrastivo', {'beta': 0.0}),
        ('wo varianza riesgo', {'alpha': 0.0}),
    ]
    rows = []
    for name, over in variants:
        for seed in args.seeds:
            a = argparse.Namespace(**vars(args))
            a.seed = seed
            for k, v in over.items():
                setattr(a, k, v)
            print(f"\n### {name} (semilla {seed})")
            tr, target = train(a)
            rows.append((name, seed) + tr.evaluate(target, target.test_idx))

    print("\n" + "=" * 62)
    print(f"{'variante':<22}{'MAE (s)':>12}{'RMSE (s)':>12}{'MAPE (%)':>12}")
    print("=" * 62)
    agg = {}
    for name, _, mae, rmse, mape in rows:
        agg.setdefault(name, []).append((mae, rmse, mape))
    for name, vals in agg.items():
        v = np.array(vals)
        print(f"{name:<22}{v[:,0].mean():9.1f}±{v[:,0].std():<3.0f}"
              f"{v[:,1].mean():9.1f}±{v[:,1].std():<3.0f}"
              f"{v[:,2].mean():9.2f}±{v[:,2].std():<3.1f}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p):
        p.add_argument('--city', default='porto')
        p.add_argument('--sources', nargs='*', default=[])
        p.add_argument('--cell-m', type=float, default=500.0)
        p.add_argument('--corridor-mode', default='raster',
                       choices=['raster', 'exact'])
        p.add_argument('--split', default='chronological',
                       choices=['chronological', 'ood_hour', 'ood_distance'])
        p.add_argument('--hours', nargs='*', type=int, default=[7, 8, 9])
        p.add_argument('--quantile', type=float, default=0.7)
        p.add_argument('--target-fraction', type=float, default=None)
        p.add_argument('--seed', type=int, default=0)

    p = sub.add_parser('preflight')
    p.add_argument('--data', required=True)
    p.add_argument('--n', type=int, default=20000)

    p = sub.add_parser('prepare')
    common(p)
    p.add_argument('--data', required=True)
    p.add_argument('--pois', default=None)
    p.add_argument('--roads', default=None)
    p.add_argument('--max-trips', type=int, default=None)

    for name in ('train', 'ablate'):
        p = sub.add_parser(name)
        common(p)
        p.add_argument('--epochs', type=int, default=60)
        p.add_argument('--patience', type=int, default=12)
        p.add_argument('--batch', type=int, default=128)
        p.add_argument('--target-per-batch', type=int, default=4)
        p.add_argument('--steps-per-epoch', type=int, default=200)
        p.add_argument('--dim', type=int, default=64)
        p.add_argument('--hidden', type=int, default=128)
        p.add_argument('--alpha', type=float, default=0.5)
        p.add_argument('--beta', type=float, default=0.5)
        p.add_argument('--device', default='auto')
        p.add_argument('--no-tsb', action='store_true')
        p.add_argument('--no-graph', action='store_true')
        p.add_argument('--no-stgnn', action='store_true')
        p.add_argument('--st-window', type=int, default=6)
        p.add_argument('--st-K', type=int, default=2)
        p.add_argument('--st-mask', type=float, default=0.3)
        p.add_argument('--lambda-st', type=float, default=0.3)
        p.add_argument('--graph-layers', type=int, default=2)
        p.add_argument('--graph-k', type=int, default=8)
        p.add_argument('--no-npm', action='store_true')
        p.add_argument('--no-invariant', action='store_true')
        if name == 'ablate':
            p.add_argument('--seeds', nargs='*', type=int, default=[0, 1, 2])

    a = _norm_args(ap.parse_args())
    {'preflight': preflight, 'prepare': prepare,
     'train': train, 'ablate': ablate}[a.cmd](a)


if __name__ == '__main__':
    main()