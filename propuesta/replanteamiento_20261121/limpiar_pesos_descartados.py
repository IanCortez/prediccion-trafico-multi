from base_replanteo import *
import shutil
logpath=OUT/'limpieza_pesos_descartados.json';log=read(logpath) if logpath.exists() else dict(removed=[],best_and_TransferTraj_untouched=True)
for name,trainer,model in [('descompuesto_aux1_s0','entrenar_descompuesto.py','modelo_descompuesto.py'),('memoria_cdf_s0','entrenar_memoria.py','modelo_memoria.py'),('memoria_mse_s0','entrenar_memoria.py','modelo_memoria.py'),('espacial_semimarkov_s0_v2','entrenar_espacial.py','modelo_espacial.py')]:
 run=(OUT/name).resolve();p=run/'modelo.pt';r=read(run/'resultado.json')
 if not p.exists():continue
 assert run.parent==OUT and p.name=='modelo.pt';assert r['validation']['mae_s']>=131.25671207822356 and r['global_goal_validation'] is False if 'global_goal_validation' in r else r['validation']['mae_s']>=131.25671207822356
 assert sha(p)==r['model_sha256'];assert (run/'validacion.npz').exists()
 assert sha(OUT/trainer)==r['code_sha256'];assert sha(OUT/model)==r['model_code_sha256']
 sources=run/'fuentes';sources.mkdir(exist_ok=True)
 for f in [trainer,model,'base_replanteo.py']:
  dst=sources/f
  if dst.exists():assert sha(dst)==sha(OUT/f)
  else:shutil.copy2(OUT/f,dst)
 rec=dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p),predictions_sha256=sha(run/'validacion.npz'),mae_s=r['validation']['mae_s'],reason='Completed non-improving experiment; predictions, metrics, protocol and exact source preserved')
 p.unlink();log['removed'].append(rec);write(run/'pesos_descartados.json',rec);write(logpath,log)
print(json.dumps(dict(removed_total_bytes=sum(z['bytes'] for z in log['removed']),experiments=[z['path'] for z in log['removed']])))

