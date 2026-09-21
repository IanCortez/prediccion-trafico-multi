"""
gbdt_tte.py
===========
Baseline de árboles para OD TTE sobre trayectorias. Reemplaza a gbdt_travel.py,
que dependía de las aristas del grafo GTFS.

Por qué sigue existiendo un baseline de árboles: TransferTraj compara contra LR
y GBM en la tarea de OD TTE, y en sus tablas GBM es competitivo frente a varias
redes. Si el modelo profundo no le saca ventaja clara, el problema no justifica
la complejidad añadida.

Recibe EXACTAMENTE las mismas columnas que el modelo neuronal (las de
tte_model.ROUTE_NAMES + CTX_NAMES, más las modalidades de las celdas origen y
destino). Si el baseline tuviera menos features, cualquier mejora medida sería
del conjunto de datos y no de la arquitectura.

Nota sobre la LSTM: el baseline secuencial anterior recorría la secuencia de
tramos entre paradas del GTFS. En OD TTE esa secuencia no existe como entrada —
la única secuencia disponible es la polilínea recorrida, que ES la etiqueta
(ver el aviso de fuga en trajectory_loader). Por eso no se porta un baseline
LSTM: no habría nada válido que darle de comer.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import tte_model as M
import trajectory_loader as tl


FEATURE_NAMES = (
    list(M.ROUTE_NAMES) + list(M.CTX_NAMES)
    + [f'o_{c}' for c in M.CELL_POI_NAMES] + [f'o_{c}' for c in M.CELL_ROAD_NAMES]
    + [f'd_{c}' for c in M.CELL_POI_NAMES] + [f'd_{c}' for c in M.CELL_ROAD_NAMES]
    + ['o_demand_q', 'd_demand_q']
)


def build_matrix(cd: M.CityData):
    """Tabla (X, y) desde un CityData. y en SEGUNDOS, no en log."""
    feats = cd.feats.numpy()
    poi, road = cd.cell_poi.numpy(), cd.cell_road.numpy()
    pat = cd.cell_pattern.numpy()
    o, d = cd.o_cell.numpy(), cd.d_cell.numpy()
    X = np.concatenate([
        feats,
        poi[o], road[o], poi[d], road[d],
        pat[o, -2:-1], pat[d, -1:],
    ], axis=1)
    return X, cd.raw_times.numpy()


def metrics(pred, true):
    err = pred - true
    return (float(np.abs(err).mean()),
            float(np.sqrt((err ** 2).mean())),
            float((np.abs(err) / np.maximum(true, 1)).mean() * 100))


def run(target_csv, source_csvs=(), target_name=None, source_names=None,
        target_contexts=None, source_contexts=None, cell_m=500.0,
        split='chronological', split_kwargs=None, target_fraction=None,
        pool_sources=True, verbose=True, **loader_kwargs):
    """
    Entrena el baseline. `pool_sources=True` apila los viajes de las ciudades
    fuente con los de la objetivo, que es el equivalente ingenuo del
    entrenamiento conjunto de TSJT: sin filtrado de gradiente, las fuentes
    simplemente dominan por volumen. Sirve como punto de comparación para
    aislar qué aporta el mecanismo TSB y no solo los datos extra.
    """
    source_names = source_names or [None] * len(source_csvs)
    source_contexts = source_contexts or [None] * len(source_csvs)

    tgt = tl.CityTrips(tl.load_trips(target_csv, city_name=target_name,
                                     verbose=verbose, **loader_kwargs),
                       target_name or 'target', 0, cell_m, target_contexts, verbose)
    srcs = [tl.CityTrips(tl.load_trips(p, city_name=n, verbose=verbose,
                                       **loader_kwargs),
                         n or f'source{k}', k + 1, cell_m, c, verbose)
            for k, (p, n, c) in enumerate(zip(source_csvs, source_names,
                                              source_contexts))]

    dm, ds, tm, ts = M.global_norms([tgt] + srcs)
    target = M.build_city_data(tgt, 0, dm, ds, tm, ts, split, split_kwargs)
    if target_fraction is not None:
        k = max(int(len(target.train_idx) * target_fraction), 8)
        target.train_idx = target.train_idx[:k]

    Xt, yt = build_matrix(target)
    X_tr = [Xt[target.train_idx]]
    y_tr = [yt[target.train_idx]]
    if pool_sources:
        for i, c in enumerate(srcs):
            sd = M.build_city_data(c, i + 1, dm, ds, tm, ts, 'chronological')
            Xs, ys = build_matrix(sd)
            X_tr.append(Xs[sd.train_idx])
            y_tr.append(ys[sd.train_idx])

    X_tr = np.concatenate(X_tr)
    y_tr = np.concatenate(y_tr)

    # objetivo en log: el tiempo de viaje es fuertemente asimétrico a la derecha
    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=8,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        random_state=0)
    model.fit(X_tr, np.log(np.maximum(y_tr, 1.0)))

    out = {}
    for name, idx in (('val', target.val_idx), ('test', target.test_idx)):
        if not idx:
            continue
        pred = np.exp(model.predict(Xt[idx]))
        out[name] = metrics(pred, yt[idx])
    if verbose:
        for k, (mae, rmse, mape) in out.items():
            print(f"[GBDT/{k}] MAE={mae:7.1f}s  RMSE={rmse:7.1f}s  MAPE={mape:5.2f}%")
    return model, out, target


if __name__ == '__main__':
    import urban_context as uc
    paths, ctxs = [], []
    for i, n in enumerate(['Porto', 'Chengdu']):
        p = tl.write_synthetic_csv(f'/tmp/_gb_{n}.csv', n, n_trips=1500, seed=i,
                                   center=(41 - i * 5, -8 + i * 40))
        paths.append(p)
        c = tl.CityTrips(tl.load_trips(p, city_name=n, verbose=False), n, verbose=False)
        ctxs.append(uc.synthetic_context(c.cells, seed=i, city=n))
    run(paths[0], paths[1:], target_name='Porto', source_names=['Chengdu'],
        target_contexts=ctxs[0], source_contexts=ctxs[1:],
        target_fraction=0.1, verbose=True)