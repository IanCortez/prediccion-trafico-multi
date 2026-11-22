# Resultado operativo del 21 de noviembre de 2026

El objetivo MAE <73,5 s no se alcanzo. La prueba final permanece cerrada.
Mejor MAE del protocolo primario: 131.256712 s. Brecha pendiente: 57.756712 s.

| Experimento | Protocolo | Viajes | MAE (s) |
|---|---|---:|---:|
| Referencia conservada | Primario OD estatico | 127941 | 131.256712 |
| Ajuste directo MAE: mejor epoca entrenada | Primario OD estatico | 127941 | 131.499756 |
| Adaptador control_capacidad: mejor epoca entrenada | Primario OD estatico | 127941 | 132.924707 |
| Adaptador rutas_globales: mejor epoca entrenada | Primario OD estatico | 127941 | 132.987197 |
| Modelo conjunto nuevo desde cero | Primario OD estatico | 127941 | 131.399512 |
| Regional objetivo solo | Pocas etiquetas, solo zona este | 49434 | 146.118630 |
| Regional transferencia selectiva | Pocas etiquetas, solo zona este | 49434 | 146.118630 |
| Estado causal local 2h | Online con etiquetas de viajes ya completados; diferente del estatico | 127941 | 130.418913 |

Los adaptadores y el ajuste directo usan el modelo original como retorno: su resultado seleccionado coincide con el original cuando ninguna epoca lo mejora. La tabla muestra el mejor resultado realmente entrenado, evitando atribuirles el resultado del retorno.

El modelo conjunto nuevo se entreno desde cero con 237 caracteristicas, ocho componentes de duracion y supervision geometrica auxiliar solo de entrenamiento. Agrega una ruta global con costes GPS historicos y preferencia por enlaces respaldados por entrenamiento. Se excluye el pliegue semanal de la consulta durante entrenamiento. Conserva todas las consultas de validacion, incluso si falla el calculo de ruta; en ese caso usa una representacion de respaldo. Selecciono la epoca 48 de 63; el resultado final es float32 con lote canonico 2048.

La prueba geometrica inicial mejora 6,6% al control mas simple. Frente a las rutas que ya recibia el modelo, la diferencia es pequena y su intervalo incluye cero. Mejor geometria no garantiza mejor tiempo. No se afirma haber implementado el entrenamiento neuronal decision-focused del articulo de referencia.

El estado causal admite solo viajes ya completados, con una demora de publicacion de 60 segundos; supera una prueba de invariancia a etiquetas futuras. Es un protocolo diferente y su 130,42 s no certifica la meta original.

Las comparaciones usan validacion previamente reutilizada: son exploratorias. No se cambiaron filtros, unidades, etiquetas ni cobertura para alcanzar la cifra. No se obtuvo la cohorte publicada de TransferTraj. Estos resultados no establecen una cota de imposibilidad.

Se conservan codigo, configuraciones, historias, metricas, diagnosticos y predicciones distintas. Los intermediarios descartados y las predicciones identicas se identifican en limpieza_plan.json y limpieza_resultado.json. El modelo original y las diez predicciones TransferTraj conservan sus SHA256.

El entorno sigue siendo el Conda local documentado en ../../auditoria. El modelo principal conservado se reproduce con ../reproducir_mejor.py segun ../REPRODUCIR.md. Los scripts de esta carpeta registran sus configuraciones y rechazan sobreescribir una corrida terminada. Para repetir un experimento, debe asignarse una carpeta de salida nueva dentro del proyecto; los recursos originales y las fuentes congeladas siguen intactos.

La relajacion ideal de mezclas entre el modelo original y el nuevo da 123,835791 s en Validacion. Solo descarta alcanzar 73,5 s con mezclas convexas de esos dos predictores fijos, aun con seleccion privilegiada. No es un predictor admisible ni una cota para otros modelos. Detalle: cota_mezcla_modelos_fijos.json.

La verificacion final del modelo conservado reprodujo exactamente las 127.941 predicciones, 65 consultas OD crudas, orden inverso y consulta individual. No se evaluo Test. Los pesos de la variante descartada y sus caches de rutas se eliminaron; sus predicciones y metricas permanecen. Los dos archivos de predicciones duplicadas se reemplazan por la referencia al original en sus resultados.
