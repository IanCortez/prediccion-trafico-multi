# Arquitectura y estado de la propuesta

Actualizado el 22 de noviembre de 2026. El estado de ejecución se consulta en `replanteamiento_20261121/CONTINUIDAD_OPERATIVA.json`; los protocolos y resultados de cada ensayo son la evidencia primaria.

## Contrato común

La consulta contiene únicamente origen, destino y fecha de salida: `o_lon`, `o_lat`, `d_lon`, `d_lat`, `timestamp`. No aporta trayectoria real, duración, cantidad de puntos, conductor ni tipo de llamada. Se mantienen la cohorte primaria, sus tiempos originales y la partición temporal: 1.023.485 consultas Train después de la purga por llegada, 127.941 de Validation. Test sigue cerrado.

Las 216 características de referencia combinan geometría OD, calendario de Lisboa, contexto urbano, rutas calculadas y perfiles obtenidos con Train. La disponibilidad histórica de parte del contexto estático heredado no está certificada: la evaluación es retrospectiva por lotes. La cohorte publicada de TransferTraj tiene 323.481 viajes preprocesados y no está disponible aquí; 73,5 s sigue siendo una meta externa, no una comparación experimental equivalente.

## Mejor modelo verificado: 129,990034 s

El mínimo de Validation corresponde a atencion_subtrayectos_s0, época final fija 6. Incorpora ocho componentes de atención sobre subtrayectos a un padre congelado de 30 componentes, atencion_historia_individual_s0, época 3. Ese padre requiere historica_fresca_s0, época 27, de 22 componentes. Se conservan todos los antecesores.

La distribución final contiene 38 componentes y 116 tiempos entre 60 y 1.785 s. El bloque de 22 combina ocho modos generales, ocho hipótesis públicas de ruta y seis prototipos históricos. El siguiente bloque añade ocho componentes sobre 64 viajes completos; el último añade ocho sobre 64 intervalos ordenados de viajes históricos.

Cada intervalo aporta 34 variables: siete puntos geométricos, tiempo fuente, cercanía a los extremos de consulta, calendario y contexto de comienzo/final del viaje fuente. La normalización y el anclaje temporal se ajustan con 32.768 consultas Train; las fuentes excluyen todo el grupo semanal de cada consulta Train antes de construir el índice espacial. Validation solo consulta fuentes Train cuya llegada precede a su inicio.

~~~mermaid
flowchart LR
 Q["Cinco campos de consulta"] --> F["216 características y rutas públicas"]
 Q --> H["Viajes completos de Train"]
 Q --> S["Intervalos ordenados de Train"]
 F --> P["Mezcla de 22 componentes"]
 H --> P
 P --> A["Atención histórica: 30 componentes"]
 H --> A
 A --> B["Atención de subtrayectos: 38 componentes"]
 S --> B
 B --> T["Distribución y mediana previstas"]
~~~

La diferencia frente al padre de 130,158042 s es −0,168007 s; el intervalo exploratorio por día es [−0,222310; −0,110332]. Es un cambio pequeño y no cumple el criterio predeclarado de mejora práctica de 0,5 s. RMSE pasa de 209,353414 a 209,651248 s: empeora ligeramente. No se afirma una mejora causal ni confirmación independiente.

La reproducción recargó los pesos y comprobó los cuantiles de las 127.941 consultas, más 65 consultas construidas exclusivamente con cinco campos, inversión de orden y caso individual. La evaluación final se recuperó cargando solo Validation después de superar el límite de memoria al mantener Train residente; los pesos coincidían con los de la recuperación final. No se repitió el entrenamiento ni se amplió el límite.

Para reproducirlo se necesitan la rama de 0,90 MiB, sus dos antecesores, los bancos públicos e históricos, la memoria original de puntos y los índices compactos. Las dependencias y huellas están en replanteamiento_20261121/nuevo_mejor_verificado.json y modelos_atencion_subtrayectos_protegidos.json. La receta es replanteamiento_20261121/reproducir_atencion_subtrayectos.py. Se mantienen además los demás mínimos y 43 checkpoints neuronales que cumplen el umbral de conservación.

## Progreso vial recurrente: ensayo cerrado sin mejora

Este ensayo utiliza todas las consultas primarias para el objetivo de duración y secuencias auxiliares de 894.105 viajes Train. La auditoría identificó cambios de sentido dentro de calles comprimidas: compartir un cruce no demuestra que se recorrió una calle completa. Se recuperaron posiciones parciales y se corrigieron las máscaras de supervisión; los tiempos originales no cambiaron.

Se supervisan una vez por época 41.511.860 posiciones válidas, distribuidas en 2.099.324 fragmentos de hasta 128 posiciones. Un viaje sin posiciones auxiliares útiles sigue participando en el objetivo primario. Las secuencias inferidas entre muestras GPS de 15 s no son anotaciones exactas de recorrido ni de espera.

La arquitectura utiliza representaciones de calle, contexto del vecindario, una GRU de 128 dimensiones y cuatro estilos latentes por viaje. Durante Train conoce las secuencias y relojes observados como supervisión privada. En consulta propone cuatro parejas de conexiones al mapa y recorre cuatro estilos por pareja: avanza únicamente con sus calles y tiempos previstos. La masa de conexiones omitidas conserva una salida global de respaldo.

El tiempo local es una velocidad inversa positiva multiplicada por distancia parcial, más una demora residual no negativa. El reloj absoluto se conserva en precisión de 64 bits y se contrasta con la zona horaria de Lisboa. La llegada incluye una sola vez el tramo terminal parcial y congela el estado de los recorridos terminados. El límite público de 192 pasos es fijo y no utiliza la longitud real de la consulta.

Tras seis épocas fijas se congelan los pesos. Se generan 30 resúmenes de recorridos previstos para todas las consultas Train y Validation, sin proporcionarles rutas reales. La etapa final compara un regresor de 1.000 árboles y profundidad 8 con 246 características contra un control idéntico con las 216 originales. Se guardan también los cuantiles 10, 50 y 90.

La rama directa sin recorridos generados obtuvo **137,894576 s**, por detrás del mejor modelo. Ese resultado ya está conservado; la propuesta completa obtuvo 136,111205 s frente a 135,988501 s de su control. No pasó el criterio de mejora. No se eligen nuevas épocas, estilos o límites a partir de este diagnóstico.

## Resultados recientes y replanteamiento

La consistencia del tiempo agregado se probó por separado: ajustó tiempos locales con todos los segmentos útiles de Train, manteniendo congeladas la política y la representación recurrente. El error del diagnóstico privado con recorrido conocido bajó de 95,36 a 76,29 s en 512 consultas Train. En las 127.941 consultas primarias, con recorrido generado y una etapa final idéntica, obtuvo **135,996132 s**, frente a 136,111205 s sin esa corrección y 135,988501 s del control de 216 variables. La mejora privada no se trasladó a la tarea OD.

Se probó también una alternativa global sin generación secuencial: difusión conjunta de la duración y una firma geométrica de 90 dimensiones. Recibió las mismas 216 características públicas y conservó todos los tiempos originales. Tras 30 épocas fijas, 32 pasos inversos y 64 muestras por consulta, obtuvo **139,953394 s**. No se eligieron otras épocas, cantidades de muestras o dimensiones después del resultado. Es una adaptación propia; no se presenta como reproducción de DOT.

La nueva memoria conserva los 42.624.621 puntos GPS de todos los padres Train autorizados en 241,76 MiB. Para una consulta busca puntos próximos al origen y destino, intersecta sus viajes y exige que el origen histórico preceda al destino al menos cuatro muestras. Utiliza el tiempo observado **15 × (j − i)** de cada fuente, ajustado únicamente por la relación de distancias OD. No conoce el recorrido, número de muestras o duración de la consulta.

El kernel directo con respaldo obtuvo **163,945881 s**; la mezcla con confianza según cantidad efectiva de fuentes y cercanía obtuvo **130,959974 s**. Ambos fallaron frente a 130,158042 s. La segunda fórmula queda conservada íntegramente porque cumple el umbral de conservación de 131,06 s, aunque no reemplaza al mejor modelo. Un 34,30 % de las consultas de Validation no encontró fuentes con este procedimiento. Se conservan también sus predicciones y métricas completas.

**Rama de atención ya evaluada:** aprende con Train el ajuste entre tiempos de subtrayectos y viajes completos; obtuvo el mínimo numérico de 129,990034 s descrito arriba. Recibe 64 fuentes, cada una con 34 variables: geometría de siete puntos intermedios, tiempo histórico, diferencias entre extremos, calendario y contexto de comienzo/final del viaje fuente. Ese contexto permite comprobar la hipótesis de que un intervalo interno no representa directamente la distribución de un viaje completo.

El mejor modelo queda congelado como padre de 30 componentes. Se añaden ocho componentes de atención, con normalización y anclaje temporal obtenidos exclusivamente de 32.768 consultas Train elegidas antes del entrenamiento. Se fijan seis épocas; se evalúa el estado final, sin escoger una época de Validation. Las fuentes de cada consulta Train excluyen su grupo semanal **antes** de construir el índice espacial, para que tampoco la cobertura de los vecinos dependa de la geometría excluida. Los tokens se reconstruyen al consultar: se guardan únicamente referencias compactas a padres y posiciones, sin bancos de distribuciones ni pares de puntos cuadráticos.

**Entrenamiento conjunto desde cero: cerrado.** La rama anterior aprendió con un padre que ya había visto todas las etiquetas Train. Se comprueba ahora si entrenar todo el sistema desde el inicio permite aprovechar mejor fuentes obtenidas fuera del grupo semanal. Esto es una hipótesis de optimización, no una causa demostrada del error.

Se inicializan de nuevo todos los pesos neuronales y se entrenan conjuntamente los 22 componentes de rutas/prototipos y ocho de subtrayectos. No se cargan pesos preentrenados. Se mantienen los mismos datos y fuentes y se fijan 24 épocas con descenso cosenoidal del paso de aprendizaje. La selección usa únicamente el estado final. Antes de entrenar se comprueban gradientes del bloque general y del nuevo experto, además de la memoria real de una consulta de 2.048 casos. Al evaluar se liberan los tensores Train. No se construyen bancos de distribuciones.

## Corrección semántica del prior geométrico

El entrenamiento conjunto terminó sus 24 épocas en **130,685980 s** y no mejoró 129,990034 s. Su estado completo ocupa 18,39 MiB y se conserva por estar por debajo de 131,06 s. La reproducción recargó ese checkpoint y comprobó todos los cuantiles y las consultas crudas. El control condicional no se activó.

Una auditoría independiente identificó una inconsistencia: la supervisión geométrica original usa siete posiciones separadas por distancia de arco; la nueva fuente usaba posiciones separadas por tiempo como referencia geométrica. En 512 padres Train, el error mediano entre esas posiciones fue de 212,32 m. La longitud aproximada con nueve puntos también subestimaba un 10,75 % en la mediana. Esto no invalida el MAE del procedimiento congelado, pero impone a la red una corrección auxiliar innecesaria.

El muestreo corregido por arco coincide con la supervisión original con error mediano de 0,048 m y máximo de 0,973 m tras la representación float16. Se contrastaron además 892 intervalos parciales y se comprobó que añadir muestras estacionarias cambia el tiempo histórico, manteniendo la geometría. La distancia acumulada de Train ocupa 144,72 MiB; no se alteró la memoria original de puntos.

**Ensayo cerrado: 129,994490 s, sin superar el mínimo vigente.** Se conserva toda la representación temporal de 34 variables y su normalización original. Se aporta por separado la geometría por arco únicamente al prior del objetivo geométrico auxiliar. No se añaden parámetros neuronales. Un instrumento con pesos no nulos verificó que la distribución temporal es idéntica con pesos fijos y que sí cambia la referencia geométrica.

La comparación usa el mismo padre congelado de 30 componentes, idéntica inicialización neuronal, optimizador, orden de lotes y seis épocas fijas del control de 129,990034 s. Así se comprueba el efecto de corregir el aprendizaje auxiliar sin eliminar la información temporal histórica. No se alimenta la función corregida a los pesos antiguos.

La variante preparada que reemplazaba las 34 entradas por geometría de arco no llegó a ejecutarse y fue sustituida antes de evaluar un resultado. El ensayo vigente es prior_arco_subtrayectos_s0. Todas las consultas primarias, tiempos, filtros y campos públicos siguen iguales; Test permanece cerrado.

[TrajRoute, MDM 2025](https://www.johnkrumm.net/_files/ugd/3e3382_9d657170e9144dcb873791573d75c542.pdf) aporta un antecedente para seguir y combinar trayectorias históricas. Su evaluación de rutas contra Azure Maps y su ventana temporal próxima a la consulta no equivalen a nuestro pronóstico OD retrospectivo con Train cerrado. La propuesta actual se limita a intervalos históricos completos; no se afirma haber reproducido su buscador de rutas ni sus resultados.

## Instrumentos y criterio de aceptación

- Verificación independiente de todas las posiciones, transiciones y sumas de tiempos; ninguna consulta primaria eliminada.
- Pruebas neuronales de llegada, tiempo terminal positivo contado una vez, estado final inmóvil y consistencia al cambiar el orden o tamaño del lote.
- Comparación del calendario del método real con 1.070 fechas, incluidos cambios de horario de verano y fracciones de segundo; error observado cero.
- Pruebas de gradientes y memoria antes de entrenar: una sola GPU pesada; lote Train 512 y consulta 1.024.
- Checkpoints sin la matriz pública de distancias de 2 GB: el estado neuronal ocupa 8,55 MiB.
- Comparación final con el mejor protegido y con el control. Se exige al menos 0,5 s de mejora y un intervalo exploratorio por día completamente favorable para pasar al control que elimina la memoria del prefijo.
- Un candidato nuevo requiere reproducción completa y consultas crudas antes de promoverlo. Ninguna mejora de Validation acredita por sí sola la meta de confirmación inferior a 73,5 s.

Los riesgos principales son el desfase entre entrenamiento con recorrido conocido e inferencia con recorrido generado, la aproximación de los tiempos auxiliares y la diferencia entre un intervalo interno y un viaje completo. La consistencia agregada ya se probó y no resolvió el error de la tarea OD. Los diagnósticos separan estas posibilidades; no atribuyen automáticamente un fallo al número de épocas.

Se conservan métricas, predicciones, configuración y fuentes de todos los experimentos, incluso con MAE mayor de 131,06 s. Se protegen los pesos completos de los candidatos inferiores a ese umbral, los mejores históricos y sus dependencias. Los datos originales y los resultados TransferTraj permanecen intactos.


<!-- resultado_landmarks_red_Train_v2_s0 -->
## Referencias topológicas públicas: piloto cerrado

Se codificaron cuatro perfiles dirigidos hacia y desde 32 cruces públicos de referencia; se contrastaron con la misma cantidad de variables euclidianas. El piloto usó solo Train y mantuvo los cinco campos públicos de consulta. MAE: 132,424078 s frente a 132,286900 s del control, sin cumplir el criterio de mejora. No se inicia una búsqueda de cantidad de referencias ni de profundidad de árboles.


<!-- piloto_transferencia_intervalos_Train_s0 -->
## Transferencia de intervalos a viajes completos: piloto cerrado

La supervisión auxiliar anterior actualizó un codificador preentrenado con una cabeza separada y un peso fijo; no mejoró el resultado. Este piloto cambia el mecanismo: aprende particiones compartidas de árboles desde cero con 113.830 intervalos exactos de los mismos 40.000 padres de ajuste. Distingue si cada extremo coincide con el del viaje padre, corrige de forma acotada las diferencias de covariables mediante clasificación de dominio fuera de grupo, y reserva el 20 % de peso relativo para las fuentes. La consulta de un viaje completo siempre lleva ambos indicadores de extremo activos; no aporta longitud de secuencia ni duración.

El control conserva exactamente los ejemplos, indicadores y pesos auxiliares, sustituyendo sus tiempos observados por predicciones del modelo sin aumento. Así se separa la información de los tiempos históricos de la regularización por más ejemplos. La corrección de densidad no garantiza eliminar diferencias de comportamiento entre un intervalo y un viaje completo; esa es precisamente la hipótesis sometida a prueba.

Se evalúan las mismas 10.000 consultas cronológicas reutilizadas de Train. Antes de ajustar, las 206 variables públicas coincidieron exactamente con las existentes en 17 consultas crudas, también al invertir el orden y consultar un solo caso; se comprobó que la geometría vial no depende del perfil histórico excluido. No se utiliza la trayectoria de ninguna consulta reservada. Se exige una mejora de al menos 5 s frente al modelo sin aumento y la referencia almacenada, y 2 s frente al control con tiempos artificiales, con intervalos exploratorios por día favorables. Si falla, no se busca otro peso ni cantidad de intervalos.

La motivación del submuestreo aparece en ProbETA, publicado en IEEE TITS 2025 y con revisión de autores de enero de 2026: https://arxiv.org/html/2407.05895v2. Ese modelo utiliza recorridos y viajes completados como contexto; este piloto OD es una adaptación propia y no una reproducción comparable de sus resultados.


<!-- hallazgo_busqueda_ordenada_Train -->
## Búsqueda conjunta de extremos: mecanismo geométrico verificado

Se auditaron 278.941 padres adicionales anteriores al corte Train, con 22.161.411 puntos, sin guardar otro banco GPS. En 512 consultas Train, añadirlos con el mismo presupuesto de 2.048 vecinos por extremo empeoró la cobertura: se perdieron 40 consultas y se ganaron 15. Las consultas sin fuente aumentaron del 31,84 % al 36,72 %. No se amplía el banco por volumen de datos.

El mecanismo limitante es la intersección de dos listas truncadas por separado. El nuevo instrumento utiliza esas listas para descubrir la unión de padres y busca dentro de cada uno el intervalo ordenado que minimiza la distancia a ambos extremos, con duración de fuente de 60 a 1.785 s y las mismas restricciones físicas. Una ventana móvil con respaldo exacto evita enumerar y guardar todos los pares posibles.

Con el banco original, la proporción con una fuente a menos de 100 m de error combinado pasó de 67,7734 % a 98,8281 %. Se conservaron todas las coincidencias anteriores y ningún mínimo geométrico empeoró. La verificación independiente por enumeración coincidió; invertir el orden y consultar un caso individual dio los mismos valores. Repetir el instrumento manteniendo exactamente la máscara geográfica original produjo archivos de resultados idénticos.

Esta evidencia justifica probar un constructor más eficiente y una predicción temporal Train sin filtración de etiquetas reservadas. No demuestra una mejora de MAE y no activa directamente otro entrenamiento primario. La siguiente etapa y su criterio de descarte están fijados en replanteamiento_20261121/plan_busqueda_ordenada_fuentes.json. El mejor MAE permanece en 129,990034 s y Test sigue cerrado.


<!-- resultado_transferencia_intervalos_Train_s0 -->
El aumento con ajuste de dominio terminó sin mejora: 133,785328 s frente a 133,511432 s sin aumento. Se descarta esa receta sin cambiar sus pesos o cantidad de ejemplos. El constructor ordenado ya pasó además una verificación independiente de posiciones y ordenación de las 64 fuentes, igualdad con y sin poda segura, consulta cruda y exclusión semanal previa al árbol. La nueva prueba temporal usa el banco original, sin aumento de datos: distribuciones de tiempos por tipo de extremo y un control de igual dimensión con tiempos derivados solo de geometría a 25 km/h.


<!-- resultado_cdf_ordenada_Train_s0 -->
## Señal temporal del nuevo buscador

El instrumento temporal terminó en 130,955613 s en 10.000 consultas Train, frente a 133,121328 s sin memoria y 133,210289 s con el control geométrico de igual dimensión. La información de tiempos históricos aporta una diferencia exploratoria favorable de 2,25 s frente al control. No alcanza el criterio de 5 s para ampliar esta representación comprimida y no se evalúa en Validation primaria. El siguiente diseño debe conservar la relación individual entre geometría, contexto de extremos y tiempo de cada fuente, en lugar de perderla al resumir todo en distribuciones. Ese diseño requiere otra prueba Train con el mismo criterio exigente; no se reduce el umbral después del resultado.
