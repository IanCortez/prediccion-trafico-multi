from base_replanteo import *
import shutil
cfg=read(OUT/'espacial_prior_cf_s0/protocolo.json');write(OUT/'sesiones_extra.json',dict(active=[dict(experiment='espacial_prior_cf_s0',pid=cfg['process_id'],session_id=44841)],goal_status='active',max_total_GPU_jobs=2))
p=OUT/'actualizar_estado.py';s=p.read_text(encoding='utf-8-sig');needle="pending=state.get('pending',[])";extra="pending=state.get('pending',[])\nextra_state=read(OUT/'sesiones_extra.json') if (OUT/'sesiones_extra.json').exists() else {}\nfor item in extra_state.get('active',[]):\n if not (OUT/item['experiment']/'resultado.json').exists() and item['pid'] not in [q.get('pid') for q in active]:active.append(item)"
if 'extra_state=' not in s:assert needle in s;s=s.replace(needle,extra)
s=s.replace("la dirección real de la consulta nunca se entrega.","la dirección real de la consulta nunca se entrega. Se ejecuta también una corrección neuronal sobre un prior de conteos locales, excluyendo de esos conteos todo el bloque de la consulta Train; este prior no excluye el ajuste neuronal ni el grafo permitido entrenado con Train.")
s=s.replace("La memoria de duraciones sin base no mejora el modelo conservado.","La memoria de duraciones sin base no mejora el modelo conservado. La variante de residuos fuera de bloque terminó en 136,3058 s, frente a 136,1861 s de su base; tampoco mejora en MAE. No se vuelve a ampliar esta variante sin una hipótesis adicional.")
p.write_text(s,encoding='utf-8')
r=read(OUT/'memoria_residual_oof_s0/resultado.json');write(OUT/'conclusion_memoria_residual.json',dict(validation=r['validation'],base_validation=r['base_validation'],paired_vs_base=r['paired_vs_base'],paired_vs_retained=r['paired_vs_retained'],conclusion='Exact all-bank gated mean-residual kernel with same-fold OOF training does not improve base MAE or the protected model. Improved RMSE does not satisfy MAE goal. Do not generalize to every retrieval architecture.',Test_access=False))
logpath=OUT/'limpieza_memoria_residual.json';removed=read(logpath)['removed'] if logpath.exists() else []
for name,filename,source in [('memoria_residual_oof_s0','modelo.pt','entrenar_residual.py'),('base_oof_estatica_s0','modelo.cbm','preparar_base_oof.py')]:
 run=(OUT/name).resolve();model=run/filename;r=read(run/'resultado.json')
 if not model.exists():continue
 assert run.parent==OUT and r['validation']['mae_s']>=131.06 and sha(model)==r['model_sha256'];assert (run/'validacion.npz').exists()
 sources=run/'fuentes';sources.mkdir(exist_ok=True)
 for name2 in [source,'base_replanteo.py']:
  dst=sources/name2
  if dst.exists():assert sha(dst)==sha(OUT/name2)
  else:shutil.copy2(OUT/name2,dst)
 rec=dict(path=str(model.relative_to(ROOT)),bytes=model.stat().st_size,sha256=sha(model),validation=r['validation'],predictions_sha256=sha(run/'validacion.npz'),reason='Completed MAE not below131.06s; metrics, predictions, protocol, exact source and OOF prediction instrument retained');model.unlink();removed.append(rec);write(run/'pesos_descartados.json',rec)
write(logpath,dict(removed=removed,best_and_TransferTraj_untouched=True))
p=BASE/'RESUMEN_RESULTADOS.md';s=p.read_text(encoding='utf-8-sig');needle='| Proceso espacial semi-Markov sin dirección | 138,223439 s |';s=s.replace(needle,needle+'\n| Memoria de residuos fuera de bloque | 136,305803 s |');s=s.replace('Siguen en ejecución una memoria de residuos con predicciones fuera de cada bloque y un proceso espacial cuyo estado conserva dirección de llegada.','La memoria de residuos tampoco mejoró su base (136,186084 s). Siguen en evaluación el proceso espacial con dirección y una variante con un prior de transiciones empíricas y corrección neuronal.');p.write_text(s,encoding='utf-8')
print(json.dumps(dict(extra_active=extra_state if 'extra_state' in globals() else cfg['process_id'],residual_result=r.get('validation'),removed_bytes=sum(z['bytes'] for z in removed))))

