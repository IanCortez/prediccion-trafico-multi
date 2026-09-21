"""
compat.py
=========
Diferencias entre Windows y POSIX que afectan a este pipeline. Todos los módulos
importan de aquí en lugar de llamar a `open` directamente.

Los cuatro problemas que resuelve, por orden de probabilidad de aparecer:

1. CONSOLA. La consola de Windows usa cp1252 por defecto y este código imprime
   flechas, ± y caracteres de caja. Sin arreglarlo, el primer print con "→"
   revienta con UnicodeEncodeError y el script muere DESPUÉS de haber
   preprocesado, que es el peor momento posible.

2. FINALES DE LÍNEA. Un CSV con CRLF leído sin `newline=''` hace que el módulo
   csv vea saltos de línea extra dentro de campos entrecomillados. El campo
   POLYLINE es largo y va entrecomillado, así que es exactamente el caso que
   rompe.

3. LÍMITE DE CAMPO EN CSV. `csv.field_size_limit(sys.maxsize)` es el idiom
   habitual pero lanza OverflowError en Windows, donde el límite interno es un
   C long de 32 bits. Hay que bajar el valor hasta que lo acepte.

4. RUTAS. Aceptar `pathlib.Path` además de str, expandir `~`, y no asumir `/`
   como separador.
"""

import csv
import os
import sys


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONSOLA
# ─────────────────────────────────────────────────────────────────────────────

def enable_unicode_console():
    """
    Fuerza UTF-8 en stdout/stderr. Con `errors='replace'` como red de seguridad:
    si la fuente de la consola no puede dibujar un carácter, se sustituye en
    lugar de abortar. Perder un símbolo de una tabla es aceptable; perder tres
    horas de entrenamiento por un print, no.

    Se llama al importar; en Python 3.7+ `reconfigure` existe siempre, y el
    try/except cubre stdout redirigidos que no lo soportan.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError, OSError):
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 2 y 3. CSV
# ─────────────────────────────────────────────────────────────────────────────

def set_csv_field_limit():
    """
    Sube el límite de tamaño de campo tanto como acepte la plataforma.

    El límite por defecto son 131.072 caracteres. Una polilínea de 3.000 puntos
    ronda los 66 KB, así que cabe, pero el margen es escaso y el dataset tiene
    colas largas.

    Se prueban candidatos en orden descendente en lugar de dividir por 10 en
    bucle: dividir sobrepasa el valor bueno. Partiendo de sys.maxsize
    (9.2e18) el primer paso cae a 9.2e17, sigue siendo demasiado, y para cuando
    la plataforma lo acepta se ha quedado en 9,2e8, muy por debajo del
    2.147.483.647 que Windows sí admite.
    """
    for limit in (sys.maxsize, 2 ** 31 - 1, 2 ** 30, 2 ** 27, 2 ** 24, 131072):
        try:
            csv.field_size_limit(limit)
            return limit
        except (OverflowError, ValueError):
            continue
    return csv.field_size_limit()


def open_csv(path, mode='r'):
    """
    Abre un CSV con los parámetros correctos en cualquier plataforma.

    - `newline=''` es OBLIGATORIO para el módulo csv, tanto al leer como al
      escribir. Sin él, en Windows la escritura mete un \\r de más por fila y la
      lectura parte los campos entrecomillados que contienen saltos.
    - `utf-8-sig` al leer descarta el BOM que añaden Excel y algunas
      exportaciones de Windows; sin eso la primera columna se llamaría
      '\\ufeffTRIP_ID' y el DictReader no la encontraría.
    """
    path = as_path(path)
    if 'w' in mode or 'a' in mode:
        return open(path, mode, encoding='utf-8', newline='')
    return open(path, mode, encoding='utf-8-sig', newline='')


def open_text(path, mode='r'):
    """Texto que no es CSV (JSON, GeoJSON). Con utf-8-sig al leer, por el BOM."""
    path = as_path(path)
    enc = 'utf-8' if ('w' in mode or 'a' in mode) else 'utf-8-sig'
    return open(path, mode, encoding=enc)


# ─────────────────────────────────────────────────────────────────────────────
# 4. RUTAS
# ─────────────────────────────────────────────────────────────────────────────

def as_path(p):
    """
    Normaliza a str: acepta str, bytes, pathlib.Path, y expande `~`.

    `os.fspath` es lo que hay que usar en vez de `isinstance(p, str)`: con la
    comprobación de tipo, pasar un `Path` —que es lo natural en Windows— hace
    que el código lo trate como si no fuera una ruta.
    """
    if p is None:
        return None
    return os.path.expanduser(os.fspath(p))


def is_path_like(p):
    """True si el objeto puede usarse como ruta (str, bytes o os.PathLike)."""
    return isinstance(p, (str, bytes, os.PathLike))


def ensure_dir(path):
    """Crea el directorio de un archivo. Tolera rutas sin directorio ('a.csv')."""
    d = os.path.dirname(as_path(path))
    if d:
        os.makedirs(d, exist_ok=True)
    return path


# se aplica al importar: cualquier módulo del proyecto que haga
# `import compat` ya queda con la consola y el límite de csv en orden
enable_unicode_console()
set_csv_field_limit()