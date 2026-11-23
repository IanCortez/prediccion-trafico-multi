# Guía para Ian: revisión de resultados y modelos

Instantánea de investigación del 22 de noviembre de 2026. Los archivos originales src/ se mantienen.
La rama aporta código, pesos retenidos, métricas, predicciones, protocolos y evidencia de resultados
negativos. Las búsquedas usaron Validation de forma adaptativa: las cifras no son confirmación de Test.

## Actualización de esta entrega

El nuevo mínimo completo verificado es **129,990034 s**, del modelo de atención de
subtrayectos. Ver [actualización y dependencias](ACTUALIZACION_129_99.md).
Se incluyen su checkpoint y los dos padres requeridos. Los resultados, checkpoints
y pilotos nuevos no deben confundirse con los estados históricos de la primera entrega.

## Cifras que deben distinguirse

| Sistema | MAE en segundos | Alcance |
|---|---:|---|
| Código original corregido | 140,822594 | 127.941 viajes de Validation |
| Mejor modelo actual: atención de subtrayectos | **129,990034** | Misma Validation; cadena de tres checkpoints incluida |
| Modelo original conservado | 131,256712 | Misma Validation; checkpoint incluido |
| Ensamble histórico | 131,063381 | Misma Validation; registro incluido, componentes incompletos |
| TransferTraj adaptado sin preentrenamiento | 164,367686 | Misma Validation |
| TransferTraj adaptado con preentrenamiento de 30 épocas | 171,720112 | Misma Validation |
| Variante de TransferTraj: reloj preciso y duración MAE | 158,877006 | Misma Validation |
| Nuevo modelo con rutas globales | 131,399512 | Misma Validation; no mejoró |
| Estado causal con información de viajes ya completados | 130,418913 | Protocolo online diferente |

Los 73,5 s del artículo TransferTraj corresponden a otra cohorte de Porto (323.481 viajes).
No se obtuvo esa cohorte; aquí se conservaron 1.279.412 viajes con filtros explícitos.
Estos controles locales adaptan recursos y algunas decisiones del código oficial. No son una
reproducción exacta del resultado publicado ni demuestran transferencia entre ciudades.

Un resultado con selección privilegiada u oracle utiliza la respuesta real para elegir predicciones.
No es un modelo desplegable ni un MAE conseguido. Los pilotos internos de 4.096 o 512 casos tampoco
sustituyen la evaluación de los 127.941 viajes.

## Modelos de la primera entrega, también conservados

1. propuesta/experimentos/od_mezcla8_geometria_conjunta_s0_fp32_v2/modelo.pt:
   referencia completa, 4.280.768 parámetros, 216 atributos OD, ocho componentes de duración y
   geometría auxiliar solo durante entrenamiento. Incluye las 115 fuentes congeladas originales,
   configuración, procedencia de selección, historial y predicciones.
2. propuesta/replanteamiento_20261121/espacial_fases_referencia_s0/modelo.pt:
   alternativa con permanencias por fases; 132,700776 s. No superó la referencia.
3. propuesta/replanteamiento_20261121/espacial_prior_cf_s0/modelo.pt:
   alternativa direccional con prior de conteos; 137,538866 s. No superó la referencia.
4. propuesta/replanteamiento_20261121/prefijo_nativo_s0/modelo.pt:
   generador recurrente experimental. Entrenamiento de diez épocas terminado; la evaluación
   completa estaba en curso en la primera captura. Sus resultados posteriores están en
   el registro actualizado de ese experimento; no es el nuevo mejor modelo.

Cada peso se publica como binario de Git, por debajo de 100 MiB, con SHA256 en
manifiesto_archivos.json. Los pesos descartados de TransferTraj no se incluyen porque ya no existen;
sí se incluyen sus métricas y diez archivos de predicciones originales, uno de ellos incompleto y
marcado como tal. No confundir el intento incompleto con un modelo terminado.

## Cómo revisar sin los datos completos

Desde la raíz de esta rama, con NumPy disponible:

    python -B revision_tesis/verificar_resultados.py

Recalcula el MAE del checkpoint conservado y de los controles TransferTraj terminados desde
predicciones y etiquetas de Validation. Comprueba cantidades, alineación y huellas del paquete.
No entrena, no necesita GPU, no abre Test y no descarga datos.

Para inspeccionar el peso y hacer una inferencia pequeña sobre atributos ya preparados:

    python -B revision_tesis/probar_modelo.py --device cpu

La muestra de 65 consultas conserva el orden exacto de los 216 atributos. No contiene GPS interior.
Su MAE no sustituye la evaluación completa. CPU o distinto tamaño de lote pueden diferir de la
evaluación canónica GPU; la prueba informa esas diferencias y no redefine el resultado publicado.

## Entorno y reproducción

Se recomienda un entorno conda por prefijo dentro del checkout. En PowerShell:

    $env:CONDA_PKGS_DIRS = Join-Path (Get-Location) '.runtime/conda-pkgs'
    conda create --prefix ./.conda-env python=3.11 -y
    conda run --prefix ./.conda-env python -m pip install --cache-dir ./.runtime/pip-cache -r requirements-revision.txt
    conda run --prefix ./.conda-env python -B revision_tesis/verificar_resultados.py

requirements-revision.txt cubre los dos instrumentos de revisión. El entorno histórico completo
se conserva en auditoria/dependencias-exactas.txt y conda-explicito.txt como evidencia; contiene
paquetes y rutas de la máquina original, no es un instalador portable directo.

La reproducción canónica completa y el reentrenamiento requieren los datos preparados locales,
mapas, perfiles y dependencias originales: ver propuesta/REPRODUCIR.md. Los lanzadores conservados
esperan un entorno .conda-env bajo la raíz del checkout. Los scripts históricos con rutas absolutas
E:/UTEC/Tesis/TesisIan deben adaptarse en una copia de trabajo, manteniendo los archivos congelados.
Ningún entrenamiento se inicia al clonar o ejecutar los instrumentos de revisión.

## Lectura crítica para la tesis

- La mezcla de distribuciones mejoró modestamente al predictor directo con las mismas entradas;
  no basta para justificar novedad universal o superioridad sobre TransferTraj.
- Geometría más precisa, supervisión de bajo desplazamiento y memoria de vecinos no dieron
  mejoras consistentes en el error total. Los resultados negativos y sus controles se conservan.
- Menor pérdida local de trayectorias no demuestra menor MAE de viaje generado.
- Solo hay trayectorias completas de Porto para esta evaluación; faltan pruebas entre ciudades,
  repetición entre semillas y confirmación independiente de un candidato congelado.
- El ensamble de 131,06 s fue omitido de la política de conservación anterior. Sus tres componentes
  ausentes no están en el ZIP histórico: allí se archivaron fuentes y metadatos, no pesos.
- Test permanece reservado y el objetivo <73,5 s sigue sin alcanzarse.

El archivo propuesta/resultados/historial_evidencia.zip está desduplicado por contenido.
index.json asocia cada ruta original con un objeto SHA256 dentro del ZIP; no es un modelo cargable.
Las referencias de código externo conservan sus propios avisos/licencias y su atribución.
Los registros de procesos y rutas de máquina son evidencia de la captura, no instrucciones de ejecución.
