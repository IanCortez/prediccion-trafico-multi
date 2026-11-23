from base_replanteo import np
from numba import njit
from ruteo_proyectado import collect_cost_path
@njit(cache=True)
def append_token(code, time, distance, used, c, t, d):
    if used > 0 and code[used - 1] == c:
        time[used - 1] += t
        distance[used - 1] += d
        return used
    if used >= len(code):
        return -1
    code[used] = c
    time[used] = t
    distance[used] = d
    return used + 1
@njit(cache=True)
def one_trip(a, b, edge, fraction, flags, error, u, v, length, D, conn, routing, edge_chain):
    cap = max(4096, (b - a) * 128)
    code = np.empty(cap, np.int32)
    tt = np.empty(cap, np.float64)
    dd = np.empty(cap, np.float64)
    used = 0
    ids = np.empty(4096, np.int32)
    parts = np.empty(4096, np.float64)
    unknown = 0
    bridgefail = 0
    stationary = 0
    overflow = 0
    for i in range(a + 1, b):
        if flags[i] > 0 or error[i - 1] > 100 or error[i] > 100:
            used = append_token(code, tt, dd, used, 65535, 15.0, 0.0)
            unknown += 1
        else:
            count, cost = collect_cost_path(edge[i - 1], fraction[i - 1], edge[i], fraction[i], u, v, length, length, D, conn, routing, ids, parts)
            if count < 0 or not np.isfinite(cost) or cost > 750.001:
                used = append_token(code, tt, dd, used, 65535, 15.0, 0.0)
                unknown += 1
                bridgefail += 1
            else:
                total = parts[:count].sum()
                if total <= 1e-06:
                    used = append_token(code, tt, dd, used, edge_chain[edge[i - 1]], 15.0, 0.0)
                    stationary += 1
                else:
                    for j in range(count):
                        if parts[j] <= 0:
                            continue
                        used = append_token(code, tt, dd, used, edge_chain[ids[j]], 15 * parts[j] / total, parts[j])
                        if used < 0:
                            break
        if used < 0:
            overflow = 1
            break
    if overflow:
        return (np.array([65535], np.uint16), np.array([(b - a - 1) * 15], np.float32), np.array([0.0], np.float32), b - a - 1, bridgefail, stationary, overflow)
    return (code[:used].astype(np.uint16), tt[:used].astype(np.float32), dd[:used].astype(np.float32), unknown, bridgefail, stationary, overflow)
