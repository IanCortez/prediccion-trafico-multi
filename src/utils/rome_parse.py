#!/usr/bin/env python3
"""
rome_to_trips.py
================

Convierte un fichero de pings GPS crudos con el esquema

    "", "DriveNo", "Date and Time", "Longitude", "Latitude"

(formato del dataset de taxis de Roma) al esquema de trayectorias de la
Tabla 3.1 del PFC:

    TRIP_ID (cadena) | TAXI_ID (entero) | TIMESTAMP (unix) | POLYLINE (JSON)

y aplica la depuración del Algoritmo 1 para derivar la variable objetivo.

Diferencias con Oporto y cómo se resuelven
------------------------------------------
1. NO hay identificador de viaje. `DriveNo` identifica al vehículo, no al
   trayecto. Los viajes se segmentan cuando aparece un hueco temporal mayor
   que --gap segundos, un salto imposible (velocidad > --max-jump-kmh) o una
   parada prolongada (vehículo casi inmóvil durante más de --stop segundos).

2. El muestreo es IRREGULAR (~7 s por taxi, con huecos). Como el Algoritmo 1
   asume t = (|P|-1)·Δ con Δ constante, cada segmento se remuestrea por
   interpolación lineal a una rejilla uniforme de --delta segundos. Así la
   POLYLINE resultante es homogénea con la de Oporto y la fórmula del
   Algoritmo 1 se cumple por construcción.

3. Las columnas 4 y 5 vienen ROTULADAS al revés en el fichero de Roma: la
   columna "Longitude" contiene la latitud. Ver --coord-order.

Supuesto de E/S
---------------
El fichero está ordenado cronológicamente de forma global (lo está en el
volcado original). Eso permite procesarlo en streaming con memoria constante:
solo se mantienen en RAM los viajes abiertos. Si el fichero no estuviera
ordenado, ordénalo antes:

    sort -t, -k3,3 entrada.csv > ordenado.csv

Uso
---
    python rome_to_trips.py taxi_february.csv -o trips_roma.csv
    python rome_to_trips.py entrada.csv -o salida.csv --delta 15 --gap 180

Dependencias: solo librería estándar.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterator, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# ESTRUCTURAS DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

# Un ping crudo del fichero de entrada: (unix_ts, lon, lat)
Ping = Tuple[float, float, float]

# Un vértice de polilínea, en el orden de Oporto: (lon, lat)
Vertex = Tuple[float, float]


@dataclass
class Trip:
    """Registro con el esquema de la Tabla 3.1."""
    trip_id: str                    # id único del viaje
    taxi_id: int                    # id del vehículo (DriveNo)
    timestamp: int                  # instante de inicio, unix
    polyline: List[Vertex]          # pares (lon, lat) a intervalo Δ fijo

    def as_row(self, delta: float) -> dict:
        o, d = self.polyline[0], self.polyline[-1]
        dur = (len(self.polyline) - 1) * delta
        dist = haversine(o, d)
        return {
            'TRIP_ID': self.trip_id,
            'TAXI_ID': self.taxi_id,
            'TIMESTAMP': self.timestamp,
            'POLYLINE': json.dumps([[round(x, 6), round(y, 6)]
                                    for x, y in self.polyline],
                                   separators=(',', ':')),
            # campos derivados del Algoritmo 1: (o, d, s, t)
            'LON_O': round(o[0], 6), 'LAT_O': round(o[1], 6),
            'LON_D': round(d[0], 6), 'LAT_D': round(d[1], 6),
            'DISTANCE_M': round(dist, 1),
            'DURATION_S': int(dur),
            'N_POINTS': len(self.polyline),
        }


@dataclass
class Stats:
    """Contadores de depuración, uno por regla del Algoritmo 1."""
    pings_leidos: int = 0
    pings_invalidos: int = 0
    segmentos: int = 0
    desc_pocos_puntos: int = 0      # |P| < 4
    desc_muchos_puntos: int = 0     # |P| > 960
    desc_duracion_corta: int = 0    # t < 60
    desc_duracion_larga: int = 0    # t > 14400
    desc_distancia: int = 0         # s < 200
    desc_velocidad: int = 0         # 3.6·s/t > 120
    validos: int = 0

    def report(self) -> str:
        pct = (100.0 * self.validos / self.segmentos) if self.segmentos else 0.0
        return (
            f"\n{'='*58}\n"
            f"  DEPURACIÓN DE TRAYECTORIAS (Algoritmo 1)\n"
            f"{'='*58}\n"
            f"  Pings leídos                 {self.pings_leidos:>12,}\n"
            f"  Pings descartados (parseo)   {self.pings_invalidos:>12,}\n"
            f"  Segmentos candidatos         {self.segmentos:>12,}\n"
            f"{'-'*58}\n"
            f"  Descartes |P| < 4            {self.desc_pocos_puntos:>12,}\n"
            f"  Descartes |P| > 960          {self.desc_muchos_puntos:>12,}\n"
            f"  Descartes t < 60 s           {self.desc_duracion_corta:>12,}\n"
            f"  Descartes t > 4 h            {self.desc_duracion_larga:>12,}\n"
            f"  Descartes s < 200 m          {self.desc_distancia:>12,}\n"
            f"  Descartes v > 120 km/h       {self.desc_velocidad:>12,}\n"
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

_TZ_CORTA = re.compile(r'([+-]\d{2})$')


def parse_timestamp(s: str) -> float:
    """
    Convierte '2014-02-01 00:00:00.739166+01' a unix float.

    El offset viene con dos dígitos ('+01'), que fromisoformat solo acepta a
    partir de Python 3.11; se normaliza a '+01:00' para admitir 3.8+.
    """
    s = s.strip()
    s = _TZ_CORTA.sub(r'\1:00', s)
    # fracciones de segundo con un número arbitrario de dígitos
    m = re.search(r'\.(\d+)', s)
    if m and len(m.group(1)) != 6:
        s = s.replace('.' + m.group(1), '.' + m.group(1)[:6].ljust(6, '0'))
    return datetime.fromisoformat(s).timestamp()


def detect_coord_order(path: str, n: int = 2000) -> str:
    """
    Decide si las columnas 4 y 5 son (lat, lon) o (lon, lat).

    En Roma la cabecera dice Longitude/Latitude pero el contenido es al revés.
    Heurística: la latitud está acotada a [-90, 90]; si la columna 5 se sale de
    ese rango, es la longitud y el orden es (lat, lon). Si ambas caben en el
    rango (caso de Roma: 41.9 y 12.5), se usa la mayor dispersión geográfica
    esperada y, en empate, se asume (lat, lon), que es el layout real del
    volcado de Roma.
    """
    c4, c5 = [], []
    with open(path, newline='', encoding='utf-8', errors='replace') as fh:
        for row in csv.reader(fh):
            if len(row) < 5:
                continue
            try:
                c4.append(float(row[3]))
                c5.append(float(row[4]))
            except ValueError:
                continue
            if len(c4) >= n:
                break
    if not c4:
        return 'lat,lon'
    if max(abs(v) for v in c5) > 90:
        return 'lat,lon'
    if max(abs(v) for v in c4) > 90:
        return 'lon,lat'
    return 'lat,lon'


def read_pings(path: str, coord_order: str,
               bbox: Optional[Tuple[float, float, float, float]],
               stats: Stats) -> Iterator[Tuple[int, Ping]]:
    """
    Lee el CSV crudo y emite (taxi_id, (unix_ts, lon, lat)).

    Tolera la presencia o ausencia de cabecera y las comillas del volcado.
    """
    with open(path, newline='', encoding='utf-8', errors='replace') as fh:
        for row in csv.reader(fh):
            if len(row) < 5:
                continue
            stats.pings_leidos += 1
            try:
                taxi = int(row[1])
                ts = parse_timestamp(row[2])
                a, b = float(row[3]), float(row[4])
            except (ValueError, TypeError):
                stats.pings_invalidos += 1   # incluye la fila de cabecera
                stats.pings_leidos -= 1
                continue

            lat, lon = (a, b) if coord_order == 'lat,lon' else (b, a)

            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                stats.pings_invalidos += 1
                continue
            if bbox and not (bbox[0] <= lat <= bbox[1] and bbox[2] <= lon <= bbox[3]):
                stats.pings_invalidos += 1
                continue

            yield taxi, (ts, lon, lat)


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENTACIÓN EN VIAJES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Buffer:
    """Viaje abierto de un taxi mientras se consume el flujo de pings."""
    pings: List[Ping] = field(default_factory=list)

    @property
    def last_ts(self) -> float:
        return self.pings[-1][0]


def _es_parada(pings: List[Ping], stop_s: float, radio_m: float) -> bool:
    """True si la cola del buffer lleva > stop_s segundos dentro de radio_m."""
    if stop_s <= 0 or len(pings) < 2:
        return False
    ref = pings[-1]
    t0 = ref[0]
    for p in reversed(pings[:-1]):
        if haversine((p[1], p[2]), (ref[1], ref[2])) > radio_m:
            return False
        if t0 - p[0] >= stop_s:
            return True
    return False


def segmentar(path: str, coord_order: str, gap: float, stop_s: float,
              stop_radio: float, max_jump_kmh: float,
              bbox, stats: Stats) -> Iterator[Tuple[int, List[Ping]]]:
    """
    Recorre el fichero en streaming y emite (taxi_id, pings_del_segmento).

    Se abre un segmento nuevo cuando entre dos pings consecutivos del mismo
    taxi hay un hueco > gap, la velocidad implícita supera max_jump_kmh
    (salto GPS imposible) o el vehículo ha permanecido parado > stop_s.
    """
    abiertos: Dict[int, _Buffer] = {}
    reloj = 0.0
    desde_barrido = 0

    for taxi, ping in read_pings(path, coord_order, bbox, stats):
        ts, lon, lat = ping
        reloj = max(reloj, ts)
        buf = abiertos.get(taxi)

        if buf is None:
            abiertos[taxi] = _Buffer([ping])
        else:
            pts, plon, plat = buf.pings[-1]
            dt = ts - pts
            if dt <= 0:                       # ping duplicado o desordenado
                continue
            d = haversine((plon, plat), (lon, lat))
            corta = (dt > gap or
                     3.6 * d / dt > max_jump_kmh or
                     _es_parada(buf.pings, stop_s, stop_radio))
            if corta:
                yield taxi, buf.pings
                stats.segmentos += 1
                abiertos[taxi] = _Buffer([ping])
            else:
                buf.pings.append(ping)

        # barrido periódico: cierra taxis que llevan mucho sin reportar
        desde_barrido += 1
        if desde_barrido >= 100_000:
            desde_barrido = 0
            for t in [t for t, b in abiertos.items() if reloj - b.last_ts > gap]:
                yield t, abiertos.pop(t).pings
                stats.segmentos += 1

    for taxi, buf in abiertos.items():
        yield taxi, buf.pings
        stats.segmentos += 1


# ─────────────────────────────────────────────────────────────────────────────
# REMUESTREO A Δ CONSTANTE
# ─────────────────────────────────────────────────────────────────────────────

def remuestrear(pings: List[Ping], delta: float) -> List[Vertex]:
    """
    Interpola linealmente los pings a una rejilla uniforme de `delta` segundos
    desde el primer instante. Devuelve la polilínea como pares (lon, lat).

    Con esto |P| = floor(T/Δ) + 1 y t = (|P|-1)·Δ, que es exactamente la
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


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITMO 1 — DERIVACIÓN DEL OBJETIVO Y DEPURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

MIN_PUNTOS, MAX_PUNTOS = 4, 960
MIN_DUR, MAX_DUR = 60.0, 14_400.0
MIN_DIST = 200.0
MAX_KMH = 120.0


def validar(trip: Trip, delta: float, stats: Stats) -> bool:
    """Aplica los filtros del Algoritmo 1 y actualiza los contadores."""
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

    stats.validos += 1
    return True


# ─────────────────────────────────────────────────────────────────────────────
# ORQUESTACIÓN
# ─────────────────────────────────────────────────────────────────────────────

CAMPOS = ['TRIP_ID', 'TAXI_ID', 'TIMESTAMP', 'POLYLINE',
          'LON_O', 'LAT_O', 'LON_D', 'LAT_D',
          'DISTANCE_M', 'DURATION_S', 'N_POINTS']


def convertir(entrada: str, salida: str, delta: float, gap: float,
              stop_s: float, stop_radio: float, max_jump_kmh: float,
              coord_order: str, bbox, verbose: bool = True) -> Stats:
    stats = Stats()
    with open(salida, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS)
        w.writeheader()
        for taxi, pings in segmentar(entrada, coord_order, gap, stop_s,
                                     stop_radio, max_jump_kmh, bbox, stats):
            poly = remuestrear(pings, delta)
            if not poly:
                continue
            ts0 = int(pings[0][0])
            trip = Trip(trip_id=f"{taxi}_{ts0}", taxi_id=taxi,
                        timestamp=ts0, polyline=poly)
            if validar(trip, delta, stats):
                w.writerow(trip.as_row(delta))
    if verbose:
        print(stats.report(), file=sys.stderr)
    return stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description='Pings GPS crudos → trayectorias (Tabla 3.1 + Algoritmo 1).')
    p.add_argument('entrada', help='CSV crudo: idx, DriveNo, fecha, coord, coord')
    p.add_argument('-o', '--salida', required=True, help='CSV de trayectorias')
    p.add_argument('--delta', type=float, default=15.0,
                   help='intervalo de remuestreo en segundos (def. 15, como Oporto)')
    p.add_argument('--gap', type=float, default=180.0,
                   help='hueco máximo dentro de un viaje, en segundos (def. 180)')
    p.add_argument('--stop', type=float, default=300.0,
                   help='parada que corta el viaje, en segundos (0 lo desactiva)')
    p.add_argument('--stop-radio', type=float, default=50.0,
                   help='radio en metros que define "inmóvil" (def. 50)')
    p.add_argument('--max-jump-kmh', type=float, default=200.0,
                   help='velocidad entre pings que se considera salto GPS')
    p.add_argument('--coord-order', choices=['auto', 'lat,lon', 'lon,lat'],
                   default='auto',
                   help='orden real de las columnas 4 y 5 (Roma: lat,lon)')
    p.add_argument('--bbox', nargs=4, type=float, metavar=('LAT_MIN', 'LAT_MAX',
                                                           'LON_MIN', 'LON_MAX'),
                   help='recorte geográfico opcional')
    a = p.parse_args(argv)

    orden = (detect_coord_order(a.entrada) if a.coord_order == 'auto'
             else a.coord_order)
    print(f"[i] Orden de columnas 4,5 interpretado como: {orden}", file=sys.stderr)

    convertir(a.entrada, a.salida, a.delta, a.gap, a.stop, a.stop_radio,
              a.max_jump_kmh, orden, tuple(a.bbox) if a.bbox else None)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())