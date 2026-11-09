"""
Baselines sin red neuronal, para medir cuánto aporta el modelo.

  mediana   la duración mediana del train para todos los viajes. Es el piso: con
            etiquetas muy ruidosas (Pekín sin marca de ocupado) potencia y TEMP
            pueden quedar cerca de él, y la distancia a la mediana mide cuánta
            señal hay en los datos.
  potencia  log t = β·log d + c(hora, laborable/fin de semana), ajustado por
            mínimos cuadrados sobre el train.
  TEMP      potencia + residuo medio de los viajes de train con el mismo par de
            celdas O-D y franja de 3 h; si hay menos de MIN_TRIPS, se usa el par
            (O, D) y después la celda O.
"""

import numpy as np

from tte_model import hour_and_dow

N_SLOTS = 48        # 24 h x {laborable, fin de semana}
MIN_TRIPS = 5


def _arrays(city):
    ts = np.array([t['local_ts'] for t in city.trips])      # hora local de la ciudad
    hour, dow = hour_and_dow(ts)
    return {
        'log_d': np.log([t['od_dist'] for t in city.trips]),
        'log_t': np.log([t['travel_time'] for t in city.trips]),
        'slot': hour * 2 + (dow >= 5),
        'hour': hour,
        'o': np.array([t['o_cell'] for t in city.trips]),
        'd': np.array([t['d_cell'] for t in city.trips]),
    }


def fit_power(x, y, group, n_groups):
    """y = β·x + c[group] por mínimos cuadrados, resuelto por grupos."""
    cnt = np.bincount(group, minlength=n_groups)
    xm = np.bincount(group, x, n_groups) / np.maximum(cnt, 1)
    ym = np.bincount(group, y, n_groups) / np.maximum(cnt, 1)
    xc, yc = x - xm[group], y - ym[group]
    beta = (xc * yc).sum() / (xc * xc).sum()
    c = ym - beta * xm
    c[cnt == 0] = y.mean() - beta * x.mean()
    return beta, c


def _group_mean(train_keys, train_vals, query_keys):
    """Media de train_vals por clave, o NaN si la clave tiene menos de MIN_TRIPS viajes."""
    uniq, inv = np.unique(train_keys, return_inverse=True)
    total = np.bincount(inv, train_vals)
    count = np.bincount(inv)
    pos = np.clip(np.searchsorted(uniq, query_keys), 0, len(uniq) - 1)
    ok = (uniq[pos] == query_keys) & (count[pos] >= MIN_TRIPS)
    return np.where(ok, total[pos] / np.maximum(count[pos], 1), np.nan)


def predict(city, train_idx, test_idx):
    """Predicciones en segundos de cada baseline para test_idx."""
    a = _arrays(city)
    tr, te = np.asarray(train_idx), np.asarray(test_idx)
    beta, c = fit_power(a['log_d'][tr], a['log_t'][tr], a['slot'][tr], N_SLOTS)
    power = beta * a['log_d'] + c[a['slot']]

    C = city.grid.n_cells
    levels = [(a['o'] * C + a['d']) * 8 + a['hour'] // 3, a['o'] * C + a['d'], a['o']]
    resid = a['log_t'][tr] - power[tr]
    temp = power[te].copy()
    filled = np.zeros(len(te), dtype=bool)
    for keys in levels:
        m = _group_mean(keys[tr], resid, keys[te])
        use = ~filled & ~np.isnan(m)
        temp[use] += m[use]
        filled |= use
    median = np.full(len(te), np.median(np.exp(a['log_t'][tr])))
    return {'potencia': np.exp(power[te]), 'TEMP': np.exp(temp), 'mediana': median}
