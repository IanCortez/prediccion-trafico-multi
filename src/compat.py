"""
Pequeñas diferencias entre Windows y POSIX que afectan al pipeline: codificación
de la consola, finales de línea y BOM en los CSV, y el límite de tamaño de campo
del módulo csv (las POLYLINE son largas).
"""

import csv
import os
import sys


def enable_unicode_console():
    # la consola de Windows usa cp1252 y algunos prints llevan caracteres no ASCII
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError, OSError):
            pass


def set_csv_field_limit():
    # sys.maxsize lanza OverflowError en Windows (C long de 32 bits)
    for limit in (sys.maxsize, 2 ** 31 - 1, 2 ** 27, 131072):
        try:
            csv.field_size_limit(limit)
            return limit
        except (OverflowError, ValueError):
            continue
    return csv.field_size_limit()


def as_path(p):
    if p is None:
        return None
    return os.path.expanduser(os.fspath(p))


def open_csv(path, mode='r'):
    # newline='' es obligatorio para csv; utf-8-sig descarta el BOM de Excel
    enc = 'utf-8' if ('w' in mode or 'a' in mode) else 'utf-8-sig'
    return open(as_path(path), mode, encoding=enc, newline='')


def open_text(path, mode='r'):
    enc = 'utf-8' if ('w' in mode or 'a' in mode) else 'utf-8-sig'
    return open(as_path(path), mode, encoding=enc)


enable_unicode_console()
set_csv_field_limit()
