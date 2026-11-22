from base_replanteo import *
cal=read(OUT/'calibracion_unica_fases_s0/resultado.json')
assert cal['selected_full_validation']['n']==127941 and not cal['qualified_candidates']
for p in [BASE/'RESUMEN_RESULTADOS.md',OUT/'README.md']:
    s=p.read_text(encoding='utf-8-sig')
    s=s.replace('Se realiza una única calibración declarada de antemano; no se repetirán búsquedas de pesos para esa misma rama congelada.','La calibración única declarada de antemano terminó: el peso elegido de 0,02 obtuvo 131,257416 s; ninguno de los ocho pesos mejoró la referencia. No se repetirán búsquedas de pesos para esa rama congelada.')
    s=s.replace('Se espera la salida real del proceso GPU anterior para iniciar el entrenamiento.','El entrenamiento recurrente ya está activo, después de comprobar la salida real del proceso GPU anterior.')
    marker='<!-- calibracion_completa_prefijo_activo -->'
    if marker not in s:
        s+='\n'+marker+'\nLa calibración terminó con MAE 131,257416 s para el peso elegido, frente a 131,256712 s de referencia. No hay candidato nuevo <131,06 s. El generador de prefijos está entrenando; pasó gradientes finitos con datos reales de Train en GPU y volvió a reproducir exactamente las 127.941 predicciones de referencia. Todavía no hay MAE del generador.\n\nEl inventario tras retirar la reanudación terminada fue 13,9553 GiB. El mejor modelo y los diez archivos de predicciones TransferTraj conservaron sus huellas. Ver inventario_conservacion.json; el tamaño puede crecer temporalmente con el punto de reanudación del entrenamiento activo.\n'
    p.write_text(s,encoding='utf-8')
c=read(OUT/'CONTINUIDAD_OPERATIVA.json')
c['pipeline_prefix_session']=9112
c['pipeline_prefix_supervisor_pid']=33120
c['latest_storage_snapshot']=read(OUT/'inventario_conservacion.json')
c['training_initial_GPU_check_passed']=read(OUT/'prefijo_nativo_s0/verificacion_GPU_inicial.json')
c['do_not_rerun_old_scripts']=['registrar_conservacion.py would overwrite enhanced policy','actualizar_resumen_prefijo.py requires its old active heading; one-time transition already applied','Do not resume finished phase trial; its resume checkpoint was deleted after verified completion']
write(OUT/'CONTINUIDAD_OPERATIVA.json',c)
print('Calibración concluida, prefijos activos, conservación comprobada; objetivo sigue activo.')
