from common import *
from evaluar_iteracion import eligible,read_records
records=read_records();groups={}
for r in records:
    if not eligible(r):continue
    key=r.get('trained_model_source',r['id'])
    if key not in groups or r['config'].get('inference_precision')=='float32':groups[key]=r
ranked=sorted(groups.values(),key=lambda r:r['validation']['mae_s']);pending=[]
for folder in sorted(RUNS.iterdir()):
    if not folder.is_dir():continue
    if not (folder/'ejecucion.json').exists() or (folder/'resultado.json').exists() or (folder/'resultado_preentrenamiento.json').exists():continue
    paths=[p for p in (folder/'ejecucion.json',folder/'historia.json',folder/'mejor.json',folder/'historia_preentrenamiento.json',folder/'progreso.json') if p.exists()];updated=max(p.stat().st_mtime for p in paths);item=dict(id=folder.name,last_artifact_update_unix=updated,recent_progress_10min=time.time()-updated<600,status='Sin resultado final; no implica que el proceso siga activo.')
    try:
        h=json.loads((folder/'historia.json').read_text(encoding='utf-8-sig'));best=json.loads((folder/'mejor.json').read_text(encoding='utf-8-sig'));item.update(latest_epoch=h[-1]['epoch'],latest_validation_mae_s=h[-1]['validation']['mae_s'],best_validation_mae_s=best['validation']['mae_s'])
    except (FileNotFoundError,json.JSONDecodeError,IndexError):pass
    if (folder/'historia_preentrenamiento.json').exists():
        try:
            h=json.loads((folder/'historia_preentrenamiento.json').read_text(encoding='utf-8-sig'));item.update(phase='pretraining',latest_pretraining_epoch=h[-1]['epoch'],latest_pretraining_losses=h[-1]['losses'])
        except (json.JSONDecodeError,IndexError,KeyError):pass
    pending.append(item)
ensembles=json.loads((BASE/'instrumentos/ensambles_validacion.json').read_text()) if (BASE/'instrumentos/ensambles_validacion.json').exists() else []
sealed=not any((BASE/p).exists() for p in ('CANDIDATO.json','CONFIRMACION.json','instrumentos/PRUEBA_ABIERTA.json'))
out=dict(updated_unix=time.time(),best_individual=dict(id=ranked[0]['id'],validation=ranked[0]['validation']),best_exploratory_ensemble=ensembles[0] if ensembles else None,training_configurations_finished=len(groups),pending=pending,test_confirmation_unopened=sealed,goal_achieved=False,root=str(ROOT));write_json(BASE/'ESTADO.json',out)
lines=['# Estado del desarrollo','','Meta de confirmación: MAE <73,5 s. **Todavía no alcanzada.**','','Mejor modelo individual terminado: `'+ranked[0]['id']+'`, '+f"{ranked[0]['validation']['mae_s']:.3f} s de validación."]
if ensembles:lines.extend(['',f"Mejor ensamble exploratorio: {ensembles[0]['validation']['mae_s']:.3f} s. Pesos iguales y familias declaradas; no se ajustan pesos a etiquetas de validación."])
lines.extend(['','La prueba de confirmación permanece sin abrir. Los valores son exploratorios y la validación se ha usado de forma adaptativa.','','## Configuraciones sin resultado final','','Se informa el último artefacto observado. La ausencia de resultado no acredita que un proceso siga activo.'])
for p in pending:
    progress=f"época {p['latest_epoch']}, mejor validación {p['best_validation_mae_s']:.3f} s; " if 'latest_epoch' in p else f"preentrenamiento, época {p['latest_pretraining_epoch']}; " if 'latest_pretraining_epoch' in p else ''
    lines.append('- `'+p['id']+'`: '+progress+('progreso reciente.' if p['recent_progress_10min'] else 'sin actualización en los últimos diez minutos.'))
lines.extend(['','## Conclusiones que orientan la siguiente fase','','El original completo corregido obtuvo 140,823 s. Las mejoras no resuelven la brecha hasta la meta. El MAE no se confunde con la mediana del error absoluto.','','Los mapas 2014 y 2026 extraídos con reglas iguales no mejoraron el comparador de árboles. Proyectar extremos sobre arcos y añadir historias por vía aportó una mejora modesta en XGBoost; el control TabM alcanzó 132,124 s y la mezcla vial 131,491 s. La comparación canónica de arquitectura favorece la mezcla en 0,632 s, con las mismas entradas.','','La ruta real remuestreada reduce un diagnóstico privilegiado a 103,155 s. No es entrada OD permitida ni una cota de error irreducible. Para ocho predictores fijos anteriores, cualquier mezcla convexa tiene MAE de validación de al menos 87,326 s; esto no limita nuevos modelos.','','El modelo de giros pasó el control de exclusión del propio bloque. Su exactitud de decisiones con trayectoria prefijada fue 93,23 %, frente a 85,83 % del camino más corto. Como característica adicional, no mejoró XGBoost de forma concluyente y empeoró TabM. El diagnóstico local no predice el resultado OD de la ruta completa.','','La inferencia float32 estricta desactiva TF32. Los checkpoints y resultados originales se conservan y las auditorías numéricas no cuentan como nuevos entrenamientos.','','Todos los recursos y ejecuciones se conservan en `E:\\UTEC\\Tesis\\TesisIan`. Consulte [ARQUITECTURA.md](ARQUITECTURA.md), [BITACORA.md](BITACORA.md) y [README.md](README.md).'])
(BASE/'ESTADO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(best=out['best_individual'],pending=pending,test_sealed=sealed),ensure_ascii=False),flush=True)
