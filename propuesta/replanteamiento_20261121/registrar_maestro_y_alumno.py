from base_replanteo import *
pilot=read(OUT/'piloto_maestro_probabilistico_s0/resultado.json');proof=read(OUT/'verificacion_perdida_maestro.json')
assert pilot['worth_building_full_crossfit_teacher'] and proof['passed']
c=read(OUT/'CONTINUIDAD_OPERATIVA.json')
c['teacher_CPU']=dict(experiment='maestro_cruzado_s0',pid=10844,session=3174,device='CPU',threads=2,source_frozen=True,own_fold_exclusion='All five teachers exclude entire query UTCweek fold; deterministic features only, no pretrained shared representation')
c['soft_target_student_queue']=dict(plan='plan_alumno_maestro.json',stage='Waiting for teacher CPU and request-study chain completion',control='aux_solicitud_control_s0',control_primary_and_sampling_code_literal_match=True,CPU_probability_and_gradient_gate_passed=True,additional_GPU_jobs=0)
c['current_goal_turn_evidence']=dict(classification='progress plus verified wait',evidence=['Live risk process polled through40000+queries','Independent privileged probabilistic teacher pilot completed on43591heldTrain rows with calibrated uncertainty','Exact expected-score/gradient identity and real-Train gradients passed','Full five-fold-out teacher CPU generation actually started; distinct OD student concretely prepared'])
write(OUT/'CONTINUIDAD_OPERATIVA.json',c)
marker='<!-- maestro_probabilistico_Train_y_alumno_OD -->'
text="""

<!-- maestro_probabilistico_Train_y_alumno_OD -->
Se completó un instrumento probabilístico distinto, sin representación compartida preentrenada ni uso de Validación/Test. En 43.591 viajes reservados de Train, el maestro solo OD obtuvo MAE 144,3174 s; añadir geometría observada de Train dio 110,7896 s. La NLL bajó 3,86640 → 3,61070 y el CRPS 106,5536 → 80,5827 s; ambos intervalos exploratorios por día excluyeron cero. La cobertura del intervalo del 80 % fue 80,97 %. Es un maestro privilegiado, no una solución OD ni una cota del error posible.

Se está creando un maestro cruzado completo con cinco modelos: cada fila de Train se predice excluyendo su bloque semanal completo del ajuste del maestro. Usa únicamente variables deterministas y geometría de esa fila como información privilegiada de Train; no usa número de puntos ni velocidad máxima GPS. Se guardan dos parámetros probabilísticos por fila, no bancos CDF. No se altera ninguna etiqueta original ni la cohorte de Validación.

El alumno OD preparado añade una pérdida suave que coincide, en gradiente, con el NLL+5RPS esperado bajo el maestro. La prueba exacta de valor/gradiente y una prueba con datos reales de Train pasaron en CPU, sin actualizar pesos. La trayectoria observada no entra en el forward del alumno. Se mantendrán cuatro épocas, objetivos primarios/geometría y el control ya predeclarado, cuya función primaria y muestreo son literalmente iguales. Se requiere una comprobación GPU inicial antes de aprender.

El alumno se encadena detrás del ensayo de solicitudes y del maestro CPU, respetando una GPU. Un candidato previo que ya alcance la meta tiene prioridad para reproducción/congelación/confirmación. Estos trabajos aún no tienen un MAE OD nuevo. Ver plan_alumno_maestro.json, verificacion_perdida_maestro.json, maestro_cruzado_s0 y pipeline_alumno_estado.json.
"""
for p in [OUT/'README.md',BASE/'RESUMEN_RESULTADOS.md']:
 s=p.read_text(encoding='utf-8-sig')
 if marker not in s:p.write_text(s+text,encoding='utf-8')
print('Maestro cruzado y alumno con control equivalente registrados; objetivo activo')
