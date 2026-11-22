# Análisis de la propuesta de Ian y antecedentes recientes

Fecha de revisión: 17 de noviembre de 2026. Código estudiado: commit `b7052e01f374105e12929b9b5f5b0d9a3eaf6b80`.
Este análisis utiliza la implementación actual, el contexto recuperado de la tesis y la auditoría local. No sustituye la revisión de un manuscrito actualizado que todavía no está en esta carpeta.
Las recomendaciones son hipótesis de investigación; no son mejoras de precisión ya demostradas.

## 1. Evaluación principal

El problema es pertinente: estimar tiempos origen-destino en una ciudad con pocos viajes etiquetados, aprovechando otras ciudades.
La implementación contiene decisiones razonables: características urbanas disponibles antes del viaje, parámetros compartidos, uso de entrenamiento para estadísticas temporales y supervisión auxiliar que no se exige en inferencia.

La debilidad principal está en la conexión entre problema, mecanismo y evidencia:
1. La transferencia utiliza mecanismos con antecedentes directos y todavía necesita identificar qué decisión nueva se está evaluando.
2. Se aproxima la utilidad de una fuente mediante similitud de representaciones y gradientes, sin demostrar que eso preserve la relación entre entradas y tiempos.
3. El estado histórico y la geometría del grafo tienen interpretaciones más limitadas que sus nombres sugieren.
4. Las diferencias de medición entre datasets pueden cambiar el significado de la etiqueta.
5. Los resultados disponibles no separan todavía todos esos efectos.

Corregir implementación es necesario, pero por sí solo no establece una contribución científica.

## 2. Qué problema resuelve realmente

Entrada de inferencia: origen, destino, hora de salida, atributos del viaje disponibles, contexto OSM y estadísticas derivadas del entrenamiento.
Salida: una duración escalar. La trayectoria intermedia del viaje consultado no se entrega como entrada.
Durante aprendizaje se utiliza además longitud de la polilínea como objetivo auxiliar de circuidad.

El entrenamiento necesita etiquetas objetivo en cada paso; calcula normalizaciones y estadísticas con esos viajes.
Por tanto, la formulación actual es **adaptación supervisada multifuente con pocos datos objetivo**.
No es el régimen de generalización a una ciudad nunca observada sin etiquetas objetivo. Tampoco implementa zero-shot mediante `target_fraction=0`: ese valor deja todo el entrenamiento.

La pregunta experimental debe ser: con un presupuesto objetivo fijo, ¿qué información fuente mejora la predicción frente a aprender solo con ese presupuesto?
Una arquitectura capaz de recibir grafos distintos no demuestra por sí sola que transfiera conocimiento útil.

Referencia del código: [preparación](../../codigo-ian/src/run.py), [modelo y entrenamiento](../../codigo-ian/src/tte_model.py).

## 3. Antecedentes de 2025 y 2026: qué es comparable

Se consultaron textos completos y se revisaron las secciones metodológicas y experimentales pertinentes de nueve artículos descargados.
Para OOD4TTE se contrastó su implementación pública y la revisión previa: la descarga editorial respondió 403 en esta actualización.
La búsqueda no es una revisión sistemática exhaustiva ni demuestra ausencia de otros antecedentes.
No se utilizan resúmenes de terceros para sustentar comparaciones técnicas.

| Trabajo | Publicación | Qué aporta | Uso correcto frente a Ian |
|---|---|---|---|
| [TSJT: Drawing Informative Gradients from Sources](https://ojs.aaai.org/index.php/AAAI/article/download/32102/34257) | AAAI 2025 | Entrenamiento conjunto orientado al objetivo, descomposición del gradiente fuente y banco de prompts por nodo. | Antecedente directo del mecanismo de transferencia. El artículo predice series de tráfico; debe adaptarse al mismo predictor OD para una comparación controlada. |
| [TransferTraj](https://proceedings.neurips.cc/paper_files/paper/2025/hash/9f02b7c937447b6857b21d31d5c344e6-Abstract-Conference.html) | NeurIPS 2025 | Reconstrucción de trayectorias, información espacial relativa y expertos condicionados por contexto multimodal. | Su tarea OD usa los extremos y la salida. Es un comparador central, documentando preentrenamiento, embeddings externos y recursos objetivo. |
| [OOD4TTE](https://github.com/CrazySpy/OOD4TTE_public) | KDD 2026 | Separación ruta/contexto, entornos latentes y regularización para cambios de distribución. | Antecedente de la rama de invariancia. Su entrada contiene secuencias de ruta; la tarea y las particiones no coinciden directamente. |
| [MetaTRL](https://ojs.aaai.org/index.php/AAAI/article/view/38658) | AAAI 2026 | Metaaprendizaje y parámetros compartidos y privados en representaciones de trayectorias. | Justifica comparar adaptación local. Su evaluador TTE conserva una secuencia de segmentos; no es un baseline OD directo. |
| [UNITE](https://web.comp.polyu.edu.hk/csmlyiu/conf/KDD26_UNITE.pdf) | KDD 2026 | Genera la ruta y estima tiempos por segmento de forma conjunta, reutilizando módulos para OD y ruta conocida. | Comparador OD reciente; requiere red vial, map-matching y supervisión por segmento. No es una demostración de transferencia few-shot entre ciudades. |
| [DutyTTE](https://ojs.aaai.org/index.php/AAAI/article/download/33350/35505) | AAAI 2025 | Predicción de ruta e intervalos de tiempo mediante expertos e incertidumbre. | Compara estimación OD e inspira evaluación de confiabilidad; no basta añadir intervalos para reclamar novedad. |
| [STBaT: Balancing Imbalance](https://ijcai.org/proceedings/2025/319) | IJCAI 2025 | Balancea patrones regionales y pondera su supervisión durante transferencia. | Referencia para tratar desequilibrios de fuentes. Su tarea es flujo agregado, no duración de viajes individuales. |
| [MixTTE](https://arxiv.org/abs/2601.02943) | KDD 2026, versión de autores | Combina información de rutas y segmentos, expertos y actualización incremental. | Inspira adaptación a heterogeneidad y colas; exige ruta e información dinámica, así que no corresponde comparar sus cifras como si fueran OD. |
| [MDTI](https://arxiv.org/abs/2510.05840) | WWW 2026 según manuscrito; preprint 2025 revisado 2026 | Fusión de GPS, rejilla y red vial, representación dinámica y preentrenamiento multimodal. | Referencia de representación con trayectorias; no se ha establecido aquí equivalencia de su entrada con OD. |
| [SimpleST](https://link.springer.com/article/10.1007/s00778-026-00983-7) | The VLDB Journal 2026 | Adapta modelos espaciotemporales congelados mediante prompts ligeros. | Referencia para reducir parámetros ajustables; sus errores de forecasting no son métricas OD comparables. |

### Diferencias de protocolo que cambian la interpretación

- TransferTraj usa 5000 trayectorias objetivo en su experimento de ajuste regional. Su mejor tabla general, su tabla zero-shot y su tabla few-shot responden a preguntas distintas.
- MetaTRL limita su entrenamiento objetivo a 50000 trayectorias. Su BJ corresponde a noviembre de 2015; el conversor de Ian describe T-Drive, de febrero de 2008. El nombre de la ciudad no identifica el mismo conjunto.
- El código público de OOD4TTE divide el experimento de distancia ECML con la suma de longitudes de segmentos y un umbral fijo; Ian utiliza distancia recta O-D y un cuantil. No es el mismo cambio de distribución.
- UNITE emplea un corte temporal 40/30/30 y recursos por segmento.
- La presencia de una misma métrica, como MAE, no hace equivalentes entradas, etiquetas ni presupuestos.

Fuentes de verificación adicionales: [evaluador ETA de MetaTRL](https://github.com/Xfc30/MetaTRL/blob/main/libcity/data/dataset/eta_dataset.py), [corte por distancia de OOD4TTE](https://github.com/CrazySpy/OOD4TTE_public/blob/main/data_utils/data_split_ECML_distance.py).

## 4. En qué falla o qué sigue sin demostrarse

### 4.1 El aporte está formulado como suma de mecanismos

El filtro del código proyecta el gradiente fuente sobre el objetivo y descarta su componente paralela adversa.
Ese principio aparece en las ecuaciones 9 y 10 de TSJT; su banco de patrones también tiene un antecedente directo allí.
La implementación de Ian adapta esos mecanismos a otro predictor y cambia representación, composición de lotes y optimización.
Esto debe presentarse como adaptación, con atribución y comparación contra una adaptación mínima controlada.

La rama de invariancia procede de otro problema de entrada. Integrarla no establece automáticamente que sea necesaria en OD.
Una contribución defendible tendría que identificar la limitación concreta que la integración resuelve y mostrarla mediante una intervención experimental.
Añadir un adaptador o expertos tampoco sería novedoso por sí mismo, dados los antecedentes revisados.

### 4.2 Igualar distribuciones no equivale a transferir la relación correcta

MMD acerca las distribuciones marginales de embeddings de celdas. No identifica correspondencias semánticas entre regiones ni comprueba que una misma representación tenga igual relación con la duración.

Ejemplo conceptual: fuente con 80% de regiones residenciales y objetivo con 20%, compartiendo exactamente la misma regla de duración por tipo de región.
La diferencia marginal puede ser real y compatible con transferencia perfecta. Forzar su desaparición puede mezclar regiones distintas.
Por otro lado, dos lugares igualmente densos pueden tener diferentes velocidades reglamentarias, conectividad o definiciones de viaje.

El objetivo de la representación debería preservar información útil para la predicción; hacer difícil reconocer la ciudad es un diagnóstico auxiliar, no el criterio final de éxito.
La normalización dentro de ciudad también puede borrar diferencias absolutas útiles: el percentil 90 de densidad no representa la misma cantidad física en dos ciudades.

Una referencia teórica clásica es [Zhao et al., ICML 2019](https://proceedings.mlr.press/v97/zhao19a.html), que muestra límites del alineamiento invariante cuando cambian relaciones entre dominios. Sus resultados de clasificación orientan esta cautela; no son una prueba directa de daño en la regresión de Ian.

### 4.3 El filtro actúa sobre una señal distinta de la que se quiere proteger

En `_step` la referencia objetivo incluye TTE, circuidad auxiliar, interpolación, regularizadores y MMD.
La alineación entra en esa referencia expresamente para evitar su atenuación por el filtro.
Un gradiente puede ser favorable al objetivo compuesto y perjudicial para la pérdida principal de duración.

Además, se promedian los gradientes de todas las fuentes antes de filtrar.
Ejemplo ilustrativo escalar: gradiente objetivo +1, fuente A +1, fuente B -2.
El promedio fuente -0,5 se elimina; filtrar por separado conservaría la contribución favorable de A.
Es una limitación posible del agregado, no una demostración de que esa modificación mejore en datos reales.

A esto se añaden gradientes ruidosos por pocos ejemplos y la transformación adaptativa de AdamW.
El contraejemplo numérico previo refuta una garantía general sobre la actualización, pero no demuestra transferencia negativa real.
Incluso un paso favorable en un minibatch no garantiza generalización al test.

### 4.4 El “estado de tráfico” es un proxy histórico de viaje completo

`build_st_tensors` calcula `log(distancia_OD / duración_total)` y agrega ese valor en la celda de origen y la de destino.
Eso combina longitud relativa del recorrido, velocidades de tramos, esperas y elección de ruta.
No es una medición de velocidad local de la celda.

Un viaje que utiliza una autopista puede tener un ritmo O-D alto aunque su origen esté en calles lentas.
Asignar el ritmo total a ese origen no convierte la observación en velocidad de sus calles.
El módulo también utiliza una semana típica; no observa incidentes ni estado contemporáneo del viaje consultado.

Como covariable histórica puede ser útil, pero necesita ese nombre e interpretación, además de una ablación que pruebe su valor.

En la misma cohorte del diagnóstico:
- 32 celdas y 5376 combinaciones celda-hora de semana.
- 480 viajes etiquetados de entrenamiento.
- 706 bins observados con todo el entrenamiento: 13,13%.
- Fuera de pliegue: coberturas 7,57% y 7,18%.
- Modelo original: 128721 parámetros; módulo ST: 15590.

La escasez y la capacidad no prueban por sí solas sobreajuste. En conjunto con las curvas de validación, motivan contrastar estimadores más simples y suavizados.

Evidencia: `diagnostico_estructural.json` y los resultados previos de convergencia. No hubo entrenamiento nuevo para medir estos valores.

### 4.5 Los datasets pueden representar distintos tipos de duración

Porto aporta registros de viajes. Los conversores de Beijing y Roma construyen segmentos a partir de huecos, saltos y paradas.
Se debe demostrar que esas duraciones son comparables; la misma forma de CSV no lo garantiza.

La [guía oficial de T-Drive](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/User_guide_T-drive.pdf) describe trazas por taxi y un intervalo medio cercano a 177 s.
El [formato oficial de Roma](https://crawdad.org/~crawdadarchive/roma/taxi/20140717/taxicabs/index.html) describe identificador, timestamp y posición.
Esos esquemas no proporcionan una etiqueta de ocupación equivalente a un viaje de pasajero.
Segmentar una traza continua es una decisión sobre el significado de la variable objetivo.

Además, remuestrear a 15 s no recupera el recorrido no observado.
La fiabilidad de duración y geometría debe evaluarse por separado: un viaje con límites observados puede tener duración fiable y ruta pobre.
La pérdida de circuidad necesita una máscara de calidad geométrica; no conviene descartar toda supervisión temporal solo por una frecuencia GPS baja.

Hay también un problema de esquema: los conversores no generan `CALL_TYPE` y el cargador sustituye su ausencia por C.
Un atributo desconocido pasa a significar una categoría específica de Porto. Se necesita máscara de ausencia o excluir esa variable de la representación compartida.

### 4.6 El grafo regional no describe por sí solo las rutas

El grafo combina proximidad entre celdas y flujos O-D; parte de sus conexiones es simétrica.
Un flujo conecta extremos, no representa todos los segmentos recorridos.
Dos celdas cercanas pueden estar separadas por un río o una barrera vial.
Las características de red incorporadas aparte ayudan, pero son resúmenes entre centroides.

Este grafo puede modelar semejanza funcional de regiones. Llamar “ruta” a sus embeddings requiere matizar qué información contiene.
Con pocos viajes también cambia la fiabilidad de sus flujos y de los perfiles de demanda.
La codificación de posición respecto al centro de ciudad aporta localización; no establece semejanza semántica entre ciudades.

### 4.7 Faltan controles sobre el efecto de las pérdidas

Una Huber sobre log-tiempo normalizado y MAE/RMSE en segundos son criterios distintos.
La transformación puede ayudar con colas, pero su efecto debe medirse por grupos de duración y distancia.
Expone menos directamente el coste absoluto de errores grandes. Una corrección automática de la exponencial no resolvería por sí sola el problema.

La penalización de dependencia entre ramas presupone que conviene reducir asociaciones que podrían ser predictivas.
Geometría, tipo de zona y hora interactúan de forma legítima.
Entornos latentes balanceados no constituyen prueba de factores causales ni de invariancia.
Conviene comparar con un predictor que concatene representaciones y permita interacciones, antes de mantener todas las restricciones.

### 4.8 La evidencia experimental actual es acotada

En el diagnóstico previo sin fuentes reales, con 480 train / 600 validación / 600 test:
- Original, presupuesto ampliado: MAE 200,8 s.
- Con posiciones: 198,1 s.
- GBDT adaptado: 176,6 s.
- Ley de potencia: 184,7 s.

Son señales para exigir comparadores fuertes, no un veredicto sobre transferencia.
No se probaron todas las configuraciones del alumno: las corridas anteriores no activaron `--pace` ni incluyeron fuentes reales.
La reparación de percentiles apenas alteró MAE en esa muestra; no puede atribuirse a ese defecto toda la diferencia de precisión.

El presupuesto objetivo debe incluir entrenamiento, validación, calibración y selección de fuentes.
Las trayectorias objetivo con timestamps aportan señales de duración; llamarlas “sin etiquetas” no las vuelve un recurso gratuito.
Cada comparador necesita una contabilidad de recursos de entrenamiento e inferencia.

## 5. Mejora recomendada: una hipótesis más concreta y un modelo más controlable

### Pregunta de investigación propuesta

**¿Cuándo mejora la transferencia de tiempos O-D con pocas etiquetas objetivo, y cuánto aporta distinguir la calibración local y la fiabilidad de la supervisión fuente?**

Una posible contribución es evaluar transferencia selectiva de residuos de duración con calibración local y calidad de medición explícita.
La combinación requiere demostrar su diferencia frente a antecedentes; no se afirma que todos sus componentes sean nuevos ni que no exista otro trabajo equivalente.

No intentaría demostrar superioridad universal sobre modelos con recursos distintos.
Buscaría una mejora reproducible frente a objetivo solo y a mecanismos de transferencia adaptados, en presupuestos y calidades definidos.

### Modelo inicial

```math
log T_hat = log T_base(O,D,tau) + r_shared(z_OD,tau) + r_city(z_OD,tau).
```

- `T_base`: una referencia interpretable por distancia de red/tiempo libre o ley de potencia. El código ya incluye `--pace`: debe evaluarse primero, no presentarse como idea nueva.
- `r_shared`: corrección compacta aprendida de fuentes, usando atributos físicos y semánticos homogéneos.
- `r_city`: calibración pequeña por ciudad, inicialmente sesgo y escala; después un adaptador de pocos parámetros si la evidencia lo justifica.

Se conservarían variables en unidades físicas junto a características relativas.
Se compararía congelar el encoder de fuentes frente a ajustar todo el modelo.
Un adaptador pequeño reduce el número de parámetros aprendidos con las pocas etiquetas, pero su beneficio es una hipótesis experimental.

### Transferencia por fiabilidad y compatibilidad

Mantener una contribución distinguible por fuente o por grupo de patrones, en lugar de filtrar solo su promedio.
Estimar compatibilidad con señales objetivo de entrenamiento/metavalidación cuyo presupuesto esté contabilizado.
Controlar explícitamente fuentes de baja compatibilidad y permitir un modo equivalente a objetivo solo.

Separar:
- calidad temporal y de segmentación para la pérdida principal;
- calidad geométrica para la pérdida de circuidad;
- disponibilidad real de atributos urbanos y del viaje.

La ponderación debe contrastarse con pesos uniformes y con pesos basados solo en volumen.
No asumir que una fuente densa es siempre útil ni que una fuente con GPS disperso siempre tiene malos tiempos.

### Estado histórico más sencillo

Antes de una GNN sobre 168 horas:
- comparar hora del día con laborable/fin de semana;
- suavizar estadísticas hacia un prior por tipo de región o por conjunto de fuentes;
- incorporar número de observaciones y disponibilidad como indicadores de confianza;
- usar residuos respecto a la referencia geométrica como proxy, sin llamarlos velocidad local;
- conservar fuera de pliegue y límites temporales.

El peso del promedio local podría crecer con su cantidad de observaciones.
Este mecanismo ofrece una hipótesis clara para pocas etiquetas y una ablación directa frente al módulo actual.

### Representación de red, como extensión posterior

Si los errores se concentran en pares separados por barreras o alternativas de ruta, probar atributos de unas pocas rutas candidatas calculadas con OSM.
Se pueden construir antes del viaje; no deben confundirse con la trayectoria realmente recorrida.
Requiere corregir longitudes y sentidos viales y medir coste de preprocesamiento.
No empezaría por reproducir una arquitectura completa de generación de rutas sin demostrar que ese es el cuello de botella.

### Incertidumbre como extensión posterior

Una vez que el estimador puntual sea competitivo, añadir cuantiles y evaluar cobertura e intervalo.
La calibración requiere etiquetas y debe entrar en el presupuesto.
La cobertura empírica en una ciudad no implica garantía en otra distribución.
Esta extensión debe diferenciarse de los antecedentes OD de incertidumbre.

## 6. Experimentos decisivos

### Protocolo

1. Homogeneizar qué representa un viaje y conservar procedencia, intervalo GPS y calidad de segmentación.
2. Usar una evaluación temporal fija por ciudad objetivo, independiente de todos los ajustes.
3. Presupuestos de etiquetas objetivo, por ejemplo 100, 300, 1000 y 5000, incluyendo validación y calibración.
4. Separar dos escenarios: pocas observaciones muestreadas dentro de la ventana histórica y pocos días consecutivos antes del despliegue.
5. Evaluar cada ciudad como objetivo y cada fuente por separado, además del conjunto.
6. Repetir con diferentes semillas de selección de etiquetas y de optimización.
7. Declarar si la geometría y las consultas sin etiqueta objetivo estaban disponibles de antemano.

Con solo tres ciudades, las conclusiones deben limitarse a esos dominios.
Experimentos de regiones de Porto o degradación de GPS son diagnósticos útiles, pero no sustituyen ciudades independientes.

### Matriz mínima de comparadores

| Comparación | Qué responde |
|---|---|
| Mediana, potencia, GBDT y MLP con el mismo conjunto de información | ¿La complejidad neuronal es necesaria? |
| Red compacta objetivo solo | Referencia del efecto de transferencia. |
| Mezcla de fuentes y objetivo, misma arquitectura | ¿Aportan datos adicionales sin mecanismo especial? |
| Preentrenamiento fuente y ajuste completo | ¿Basta la estrategia convencional? |
| Encoder congelado y calibración local | ¿Conviene adaptar pocos parámetros? |
| TSJT adaptado al predictor OD | ¿Aporta el filtro respecto a la mezcla? |
| Propuesta con/sin MMD | ¿Ayuda realmente el alineamiento marginal? |
| Propuesta con/sin control de calidad | ¿Se resuelve heterogeneidad de medición? |
| TransferTraj OD bajo recursos documentados | Comparador fuerte de transferencia reciente. |
| DutyTTE/UNITE en un subconjunto compatible | Contraste OD con supervisión y red vial de otra granularidad. |

MetaTRL se usa como antecedente o adaptado a un input OD explícito; sus métricas TTE originales no se trasladan directamente.
MixTTE, STBaT y SimpleST orientan mecanismos, no se incorporan como filas de MAE OD publicadas.

### Ablaciones que identificarían el mecanismo

- Base de duración sola, base + residuo compartido, base + calibración local, combinación.
- Filtrar antes y después de agregar fuentes; mismos lotes y presupuestos.
- Compatibilidad uniforme frente a compatibilidad condicionada por patrón.
- Variables urbanas absolutas, relativas y ambas.
- Estado ST actual, estado suavizado y sin estado.
- Circuidad auxiliar con y sin máscara de calidad.
- Modelo con y sin penalizaciones de invariancia, con misma capacidad.
- En una cohorte controlada, degradar frecuencia GPS manteniendo el ground truth para distinguir error de medición y cambio urbano.

### Qué medir

MAE, RMSE y mediana del error, además de errores por distancia, hora, tipo de zona y calidad.
Ganancia de transferencia = (error objetivo solo - error con transferencia) / error objetivo solo.
Reportar ganancia media, peor caso y frecuencia de transferencia negativa, no solo el mejor resultado.
Medir parámetros ajustables, tiempo, memoria y recursos de preentrenamiento externos.
Si hay intervalos: cobertura, amplitud y puntuación de intervalos.

Los intervalos deben considerar selección de etiquetas, semillas y dependencia temporal; un bootstrap de viajes con modelos fijos solo refleja incertidumbre condicionada.
La elección del método y los hiperparámetros se hace con validación, sin escoger resultados por el test.

## 7. Prioridad de trabajo

Primero: fijar la etiqueta y la procedencia de datos y reparar los problemas ya verificados.
Después: comparar `--pace`, GBDT, un MLP compacto y adaptación local bajo presupuesto común.
Solo entonces: incorporar selección de fuentes/calidad y medir su ganancia incremental.
El resto de componentes debe conservarse por evidencia y no por complejidad de la arquitectura.

Si la transferencia selectiva no mejora consistentemente objetivo solo, una conclusión rigurosa sobre cuándo falla la transferencia puede ser más defendible que una afirmación general de superioridad.
El resultado científico depende de delimitar condiciones y producir evidencia reproducible.

## 8. Archivos de soporte y límites de la revisión

- `papers/`: nueve PDFs obtenidos de fuentes oficiales o versiones de autores.
- `textos/` y `paginas/`: extracciones y páginas clave revisadas.
- `fuentes_descargadas.json`: URLs, estado de descarga, páginas y SHA-256.
- `matriz_antecedentes.csv`: tareas, entradas y papel de cada antecedente.
- `diagnostico_estructural.json`: cobertura y tamaño del modelo.
- `codigo_fuentes/`: instantáneas de código público utilizadas para verificar comparabilidad.
- `bibliografia.bib`: referencias seleccionadas.
- La auditoría empírica anterior está en `../INFORME.md` y `../resultados/`.

No se ejecutaron nuevos entrenamientos durante esta revisión bibliográfica.
La introspección de cobertura usa el entorno conda del proyecto.
Faltan las trayectorias fuente reales y el manuscrito actualizado para cerrar la evaluación de sus tablas.

