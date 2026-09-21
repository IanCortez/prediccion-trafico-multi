#!/usr/bin/env python3
"""
beijing_to_trips.py
===================

Convierte los ficheros de pings GPS del dataset T-Drive (Pekín) al esquema de
trayectorias de la Tabla 3.1 del PFC y aplica la depuración del Algoritmo 1.

Entrada: un directorio con ficheros .txt, uno por fichero, con el esquema

    taxi_id, date_time, longitude, latitude
    1,2008-02-02 15:36:08,116.51172,39.92123

Salida: un único CSV con

    TRIP_ID | TAXI_ID | TIMESTAMP | POLYLINE  (+ campos derivados o, d, s, t)

Diferencias con Oporto y Roma
-----------------------------
1. Un fichero por unidad. Cada .txt se procesa de forma INDEPENDIENTE: nunca
   se une el final de un fichero con el principio de otro. Si el fichero ya
   contiene un solo viaje, la segmentación no lo corta y emite un registro;
   si contiene la traza completa de un taxi (formato original de T-Drive),
   la segmentación por hueco lo parte en viajes. Ambos casos funcionan sin
   cambiar de modo.

2. Muestreo ~10x más disperso que en Roma (intervalos de minutos, no de
   segundos). Esto obliga a subir --gap: con el umbral de 180 s de Roma cada
   ping quedaría aislado en su propio segmento y no sobreviviría ninguno.

3. La POLILÍNEA de Pekín NO es geometría de ruta fiable. Entre dos pings
   separados 10 minutos, la interpolación inventa el trazado. Esto no
   contamina la variable objetivo, porque el Algoritmo 1 solo usa el primer
   y el último vértice para derivar (o, d, s, t), pero sí invalida cualquier
   uso de los vértices intermedios como recorrido real. La columna MAX_GAP_S
   registra el mayor hueco crudo de cada viaje para poder filtrarlo después.

4. Marcas de tiempo locales SIN zona horaria. Se aplica --tz-offset (8 h,
   horario de Pekín) para obtener un unix timestamp correcto y comparable con
   el de los otros datasets.

5. Registros duplicados consecutivos, frecuentes en T-Drive. Se descartan los
   pings con dt <= 0 respecto al anterior.

Uso
---
    python beijing_to_trips.py ./tdrive/taxi_log_2008_by_id -o trips_beijing.csv
    python beijing_to_trips.py ./datos -o trips.csv --split --jobs 8

Dependencias: solo librería estándar.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# ESTRUCTURAS DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

Ping = Tuple[float, float, float]      # (unix_ts, lon, lat)
Vertex = Tuple[float, float]           # (lon, lat), orden de Oporto


@dataclass
class Trip:
    """Registro con el esquema de la Tabla 3.1."""
    trip_id: str
    taxi_id: int
    timestamp: int
    polyline: List[Vertex]
    max_gap_s: float = 0.0             # mayor hueco crudo interpolado
    n_raw: int = 0                     # pings reales antes del remuestreo

    def as_row(self, delta: float) -> dict:
        o, d = self.polyline[0], self.polyline[-1]
        dur = (len(self.polyline) - 1) * delta
        return {
            'TRIP_ID': self.trip_id,
            'TAXI_ID': self.taxi_id,
            'TIMESTAMP': self.timestamp,
            'POLYLINE': json.dumps([[round(x, 6), round(y, 6)]
                                    for x, y in self.polyline],
                                   separators=(',', ':')),
            'LON_O': round(o[0], 6), 'LAT_O': round(o[1], 6),
            'LON_D': round(d[0], 6), 'LAT_D': round(d[1], 6),
            'DISTANCE_M': round(haversine(o, d), 1),
            'DURATION_S': int(dur),
            'N_POINTS': len(self.polyline),
            'N_RAW': self.n_raw,
            'MAX_GAP_S': int(self.max_gap_s),
        }


@dataclass
class Stats:
    """Contadores de depuración, uno por regla del Algoritmo 1."""
    ficheros: int = 0
    pings_leidos: int = 0
    pings_invalidos: int = 0
    pings_duplicados: int = 0
    segmentos: int = 0
    desc_pocos_puntos: int = 0
    desc_muchos_puntos: int = 0
    desc_duracion_corta: int = 0
    desc_duracion_larga: int = 0
    desc_distancia: int = 0
    desc_velocidad: int = 0
    desc_hueco: int = 0                # hueco crudo > --max-gap
    validos: int = 0

    def merge(self, other: 'Stats') -> None:
        for k, v in asdict(other).items():
            setattr(self, k, getattr(self, k) + v)

    def report(self) -> str:
        pct = (100.0 * self.validos / self.segmentos) if self.segmentos else 0.0
        return (
            f"\n{'='*58}\n"
            f"  DEPURACIÓN DE TRAYECTORIAS — PEKÍN (Algoritmo 1)\n"
            f"{'='*58}\n"
            f"  Ficheros procesados          {self.ficheros:>12,}\n"
            f"  Pings leídos                 {self.pings_leidos:>12,}\n"
            f"  Pings descartados (parseo)   {self.pings_invalidos:>12,}\n"
            f"  Pings duplicados             {self.pings_duplicados:>12,}\n"
            f"  Segmentos candidatos         {self.segmentos:>12,}\n"
            f"{'-'*58}\n"
            f"  Descartes |P| < 4            {self.desc_pocos_puntos:>12,}\n"
            f"  Descartes |P| > 960          {self.desc_muchos_puntos:>12,}\n"
            f"  Descartes t < 60 s           {self.desc_duracion_corta:>12,}\n"
            f"  Descartes t > 4 h            {self.desc_duracion_larga:>12,}\n"
            f"  Descartes s < 200 m          {self.desc_distancia:>12,}\n"
            f"  Descartes v > 120 km/h       {self.desc_velocidad:>12,}\n"
            f"  Descartes hueco crudo        {self.desc_hueco:>12,}\n"
            f"{'-'*58}\n"
            f"  Viajes válidos               {self.validos:>12,}  ({pct:.1f} %)\n"
            f"{'='*58}\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES GEOMÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

R_TIERRA = 6_371_000.0


def haversine(a: Vertex, b: Vertex) -> float:
    """Distancia en metros entre dos vértices (lon, lat)."""
    lon1, lat1 = a
    lon2, lat2 = b
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_TIERRA * math.asin(math.sqrt(h))


# ─────────────────────────────────────────────────────────────────────────────
# PARSEO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

FMT = '%Y-%m-%d %H:%M:%S'
BBOX_PEKIN = (39.4, 41.1, 115.4, 117.5)      # (lat_min, lat_max, lon_min, lon_max)


def leer_fichero(path: str, tz_offset: float, bbox, stats: Stats) -> Tuple[int, List[Ping]]:
    """
    Lee un .txt de T-Drive y devuelve (taxi_id, pings ordenados).

    El orden de columnas se verifica por fichero: la longitud de Pekín (~116)
    supera 90, lo que la distingue sin ambigüedad de la latitud (~39).
    """
    tz = timezone(timedelta(hours=tz_offset))
    taxi_id = -1
    pings: List[Ping] = []

    with open(path, newline='', encoding='utf-8', errors='replace') as fh:
        for row in csv.reader(fh):
            if len(row) < 4:
                continue
            stats.pings_leidos += 1
            try:
                tid = int(row[0])
                ts = datetime.strptime(row[1].strip(), FMT).replace(tzinfo=tz).timestamp()
                a, b = float(row[2]), float(row[3])
            except (ValueError, TypeError):
                stats.pings_invalidos += 1
                stats.pings_leidos -= 1          # cabecera u otra fila no numérica
                continue

            # (lon, lat) esperado; se invierte si las magnitudes lo desmienten
            lon, lat = (a, b) if abs(a) > 90 or abs(b) <= 90 else (b, a)

            if lon == 0.0 and lat == 0.0:        # ping nulo, habitual en T-Drive
                stats.pings_invalidos += 1
                continue
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                stats.pings_invalidos += 1
                continue
            if bbox and not (bbox[0] <= lat <= bbox[1] and bbox[2] <= lon <= bbox[3]):
                stats.pings_invalidos += 1
                continue

            taxi_id = tid
            pings.append((ts, lon, lat))

    pings.sort(key=lambda p: p[0])
    return taxi_id, pings


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENTACIÓN EN VIAJES (dentro de cada fichero)
# ─────────────────────────────────────────────────────────────────────────────

def segmentar(pings: List[Ping], gap: float, stop_s: float, stop_radio: float,
              max_jump_kmh: float, stats: Stats) -> Iterable[List[Ping]]:
    """
    Parte la traza de un fichero en segmentos.

    Se corta cuando hay un hueco > gap, un salto imposible (> max_jump_kmh) o
    una parada de más de stop_s segundos dentro de stop_radio metros.
    """
    actual: List[Ping] = []
    quieto_desde: Optional[float] = None

    for ping in pings:
        if not actual:
            actual = [ping]
            quieto_desde = ping[0]
            continue

        pts, plon, plat = actual[-1]
        ts, lon, lat = ping
        dt = ts - pts
        if dt <= 0:                               # duplicado o desordenado
            stats.pings_duplicados += 1
            continue

        d = haversine((plon, plat), (lon, lat))
        if d > stop_radio:
            quieto_desde = ts
        parado = (stop_s > 0 and quieto_desde is not None
                  and ts - quieto_desde >= stop_s)

        if dt > gap or 3.6 * d / dt > max_jump_kmh or parado:
            if len(actual) >= 2:
                yield actual
                stats.segmentos += 1
            actual = [ping]
            quieto_desde = ts
        else:
            actual.append(ping)

    if len(actual) >= 2:
        yield actual
        stats.segmentos += 1


# ─────────────────────────────────────────────────────────────────────────────
# REMUESTREO A Δ CONSTANTE
# ─────────────────────────────────────────────────────────────────────────────

def remuestrear(pings: List[Ping], delta: float) -> List[Vertex]:
    """
    Interpola linealmente a una rejilla uniforme de `delta` s desde el primer
    instante, de modo que |P| = floor(T/Δ)+1 y t = (|P|-1)·Δ, que es la
    hipótesis del Algoritmo 1.
    """
    if len(pings) < 2:
        return [(p[1], p[2]) for p in pings]

    t0, tN = pings[0][0], pings[-1][0]
    n = int((tN - t0) // delta) + 1
    salida: List[Vertex] = []
    j = 0
    for k in range(n):
        t = t0 + k * delta
        while j + 1 < len(pings) - 1 and pings[j + 1][0] <= t:
            j += 1
        ta, lona, lata = pings[j]
        tb, lonb, latb = pings[j + 1]
        w = 0.0 if tb == ta else (t - ta) / (tb - ta)
        w = min(max(w, 0.0), 1.0)
        salida.append((lona + w * (lonb - lona), lata + w * (latb - lata)))
    return salida


def mayor_hueco(pings: List[Ping]) -> float:
    return max((b[0] - a[0] for a, b in zip(pings, pings[1:])), default=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITMO 1 — DERIVACIÓN DEL OBJETIVO Y DEPURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

MIN_PUNTOS, MAX_PUNTOS = 4, 960
MIN_DUR, MAX_DUR = 60.0, 14_400.0
MIN_DIST = 200.0
MAX_KMH = 120.0


def validar(trip: Trip, delta: float, max_gap: float, stats: Stats) -> bool:
    P = trip.polyline
    if len(P) < MIN_PUNTOS:
        stats.desc_pocos_puntos += 1
        return False
    if len(P) > MAX_PUNTOS:
        stats.desc_muchos_puntos += 1
        return False

    t = (len(P) - 1) * delta
    if t < MIN_DUR:
        stats.desc_duracion_corta += 1
        return False
    if t > MAX_DUR:
        stats.desc_duracion_larga += 1
        return False

    s = haversine(P[0], P[-1])
    if s < MIN_DIST:
        stats.desc_distancia += 1
        return False
    if 3.6 * s / t > MAX_KMH:
        stats.desc_velocidad += 1
        return False
    if max_gap > 0 and trip.max_gap_s > max_gap:
        stats.desc_hueco += 1
        return False

    stats.validos += 1
    return True


# ─────────────────────────────────────────────────────────────────────────────
# PROCESADO POR FICHERO
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    delta: float = 15.0
    gap: float = 900.0
    stop: float = 0.0
    stop_radio: float = 50.0
    max_jump_kmh: float = 200.0
    max_gap: float = 0.0
    tz_offset: float = 8.0
    bbox: Optional[Tuple[float, float, float, float]] = None


def procesar_fichero(args: Tuple[str, Config]) -> Tuple[List[dict], Stats]:
    """Devuelve (filas, stats) de un único .txt. Apto para multiproceso."""
    path, cfg = args
    stats = Stats(ficheros=1)
    taxi_id, pings = leer_fichero(path, cfg.tz_offset, cfg.bbox, stats)
    if taxi_id < 0:
        return [], stats

    base = os.path.splitext(os.path.basename(path))[0]
    filas: List[dict] = []
    for seg in segmentar(pings, cfg.gap, cfg.stop, cfg.stop_radio,
                         cfg.max_jump_kmh, stats):
        poly = remuestrear(seg, cfg.delta)
        if not poly:
            continue
        ts0 = int(seg[0][0])
        trip = Trip(trip_id=f"{base}_{ts0}", taxi_id=taxi_id, timestamp=ts0,
                    polyline=poly, max_gap_s=mayor_hueco(seg), n_raw=len(seg))
        if validar(trip, cfg.delta, cfg.max_gap, stats):
            filas.append(trip.as_row(cfg.delta))
    return filas, stats


# ─────────────────────────────────────────────────────────────────────────────
# ORQUESTACIÓN
# ─────────────────────────────────────────────────────────────────────────────

CAMPOS = ['TRIP_ID', 'TAXI_ID', 'TIMESTAMP', 'POLYLINE',
          'LON_O', 'LAT_O', 'LON_D', 'LAT_D',
          'DISTANCE_M', 'DURATION_S', 'N_POINTS', 'N_RAW', 'MAX_GAP_S']


def listar_ficheros(raiz: str, patron: str) -> List[str]:
    if os.path.isfile(raiz):
        return [raiz]
    sufijo = patron.lstrip('*')
    out = []
    for dirpath, _, names in os.walk(raiz):
        out.extend(os.path.join(dirpath, n) for n in names if n.endswith(sufijo))
    return sorted(out)


def marcar_split(filas: List[dict], ratios: Tuple[float, float, float]) -> None:
    """
    Añade la columna SPLIT con un corte TEMPORAL, no aleatorio.

    El muestreo aleatorio filtraría condiciones de tráfico del futuro hacia el
    entrenamiento y anularía el desplazamiento de distribución que el trabajo
    quiere medir. El corte por fecha lo preserva.
    """
    filas.sort(key=lambda r: r['TIMESTAMP'])
    n = len(filas)
    c1 = int(n * ratios[0])
    c2 = c1 + int(n * ratios[1])
    for i, r in enumerate(filas):
        r['SPLIT'] = 'train' if i < c1 else ('val' if i < c2 else 'test')


def convertir(raiz: str, salida: str, cfg: Config, patron: str, jobs: int,
              split: Optional[Tuple[float, float, float]]) -> Stats:
    ficheros = listar_ficheros(raiz, patron)
    if not ficheros:
        print(f"[!] No se encontró ningún fichero '{patron}' en {raiz}",
              file=sys.stderr)
        return Stats()
    print(f"[i] {len(ficheros):,} ficheros a procesar", file=sys.stderr)

    total = Stats()
    campos = CAMPOS + (['SPLIT'] if split else [])
    tareas = [(f, cfg) for f in ficheros]

    if split:
        # el corte temporal necesita el conjunto completo antes de escribir
        acumulado: List[dict] = []
        for filas, st in _ejecutar(tareas, jobs):
            acumulado.extend(filas)
            total.merge(st)
        marcar_split(acumulado, split)
        with open(salida, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=campos)
            w.writeheader()
            w.writerows(acumulado)
    else:
        with open(salida, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=campos)
            w.writeheader()
            for filas, st in _ejecutar(tareas, jobs):
                w.writerows(filas)
                total.merge(st)

    print(total.report(), file=sys.stderr)
    return total


def _ejecutar(tareas, jobs: int):
    if jobs <= 1:
        for t in tareas:
            yield procesar_fichero(t)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            yield from ex.map(procesar_fichero, tareas, chunksize=32)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description='T-Drive (Pekín) → trayectorias (Tabla 3.1 + Algoritmo 1).')
    p.add_argument('entrada', help='directorio con los .txt (o un único fichero)')
    p.add_argument('-o', '--salida', required=True)
    p.add_argument('--patron', default='*.txt')
    p.add_argument('--delta', type=float, default=15.0,
                   help='intervalo de remuestreo en s (def. 15, como Oporto)')
    p.add_argument('--gap', type=float, default=900.0,
                   help='hueco máximo dentro de un viaje en s (def. 900; el '
                        'muestreo de T-Drive es de minutos, no de segundos)')
    p.add_argument('--stop', type=float, default=0.0,
                   help='parada que corta el viaje en s (def. 0 = desactivado)')
    p.add_argument('--stop-radio', type=float, default=50.0)
    p.add_argument('--max-jump-kmh', type=float, default=200.0)
    p.add_argument('--max-gap', type=float, default=0.0,
                   help='descarta viajes cuyo mayor hueco CRUDO supere este '
                        'valor en s (0 = no descartar, solo anotar MAX_GAP_S)')
    p.add_argument('--tz-offset', type=float, default=8.0,
                   help='huso horario de las marcas de tiempo (def. 8 = Pekín)')
    p.add_argument('--bbox', nargs=4, type=float,
                   metavar=('LAT_MIN', 'LAT_MAX', 'LON_MIN', 'LON_MAX'),
                   help=f'recorte geográfico (por defecto Pekín: {BBOX_PEKIN})')
    p.add_argument('--sin-bbox', action='store_true',
                   help='desactiva el recorte geográfico')
    p.add_argument('--jobs', type=int, default=1, help='procesos en paralelo')
    p.add_argument('--split', action='store_true',
                   help='añade columna SPLIT con corte temporal train/val/test')
    p.add_argument('--ratios', nargs=3, type=float, default=[0.7, 0.15, 0.15],
                   metavar=('TRAIN', 'VAL', 'TEST'))
    a = p.parse_args(argv)

    bbox = None if a.sin_bbox else (tuple(a.bbox) if a.bbox else BBOX_PEKIN)
    cfg = Config(delta=a.delta, gap=a.gap, stop=a.stop, stop_radio=a.stop_radio,
                 max_jump_kmh=a.max_jump_kmh, max_gap=a.max_gap,
                 tz_offset=a.tz_offset, bbox=bbox)

    convertir(a.entrada, a.salida, cfg, a.patron, a.jobs,
              tuple(a.ratios) if a.split else None)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())