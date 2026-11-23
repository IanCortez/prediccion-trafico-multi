"""Bounded preparation handoff, instrument gate, GPU trial and conditional control."""
from base_replanteo import *
import subprocess,ast
start=time.time()
for name in ['modelo_aux_subviajes.py','verificar_aux_subviajes.py','entrenar_aux_subviajes.py']:
 ast.parse((OUT/name).read_text(encoding='utf-8-sig'))
write(OUT/'plan_aux_subviajes.json',dict(
 hypothesis='Exact Train subtrip OD durations can improve shared OD encoding, while a separate auxiliary head absorbs midtrip/booking domain differences.',
 distinction='Supervised interval durations and separate task head; not masked-prefix/suffix pretraining or query path input.',
 primary_labels_and_cohort_unchanged=True,auxiliary_examples_max=500000,training='1 frozen-backbone auxiliary warmup then6fixed joint epochs, lambda.2; matched zero-weight control only if primary reference beaten.',
 control_decision='No control GPU run if no learned epoch beats original131.25671207822356; all metrics/predictions retained either way.',
 risk='Subtrip time distribution differs from full trips; tasks may conflict. No expectation/guarantee of73.5 from this mechanism.',
 Test='sealed',cache_cleanup='After a negative complete trial remove only exact feature cache <=600MiB, keeping selected queries/labels/provenance, source and reconstruction instrument.'))
while not (OUT/'subviajes_train_s0/resultado.json').exists():
 if time.time()-start>2700:raise RuntimeError('Data preparation did not finish; inspect actual process/log, do not relaunch blindly')
 time.sleep(5)
def run_script(name,extra,log):
 with (OUT/log).open('w',encoding='utf-8') as f:
  process=subprocess.Popen([sys.executable,'-B',str(OUT/name),*extra],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  write(OUT/'pipeline_subviajes_estado.json',dict(stage=name,args=extra,pid=process.pid,goal='ACTIVE',Test='sealed'))
  rc=process.wait()
 if rc:raise RuntimeError(str((name,extra,'exit',rc,'inspect',log)))
run_script('verificar_aux_subviajes.py',[],'verificar_aux_subviajes.log')
run_script('entrenar_aux_subviajes.py',['--arm','auxiliary'],'aux_subviajes_s0.log')
r=read(OUT/'aux_subviajes_s0/resultado.json');reference=read(BEST/'resultado.json')['validation']['mae_s'];results=[r];control=None
if r['best_epoch']>0 and r['validation']['mae_s']<reference:
 run_script('entrenar_aux_subviajes.py',['--arm','control'],'control_aux_subviajes_s0.log')
 control=read(OUT/'control_aux_subviajes_s0/resultado.json');results.append(control)
d=np.load(OUT/'aux_subviajes_s0/validacion.npz');last='epoch_6';best_learned=min(r['all_epoch_validation'],key=lambda q:q['validation']['mae_s'])
conclusion=dict(experiments=[dict(id=z['id'],validation=z['validation'],best_epoch=z['best_epoch'],all_epoch_validation=z['all_epoch_validation']) for z in results],
 protected_reference_mae_s=reference,best_learned_epoch=best_learned,
 final_fixed_epoch_vs_reference=paired(d['true'],d[last],d['epoch_0'],d['timestamp']),
 conditional_control_run=control is not None,learned_improvement=r['best_epoch']>0,
 interpretation='One predeclared exact-subtrip auxiliary configuration; selected epoch0 is original reference, not a learned gain. Do not claim general impossibility from a negative result.',
 primary_query_unchanged=True,Test_access=False,goal='ACTIVE, not complete',seconds=time.time()-start)
if control is not None:
 c=np.load(OUT/'control_aux_subviajes_s0/validacion.npz')
 assert np.array_equal(c['row'],d['row']) and np.array_equal(c['true'],d['true'])
 conclusion['matched_fixed_epoch_aux_minus_control']=paired(d['true'],d[last],c[last],d['timestamp'])
write(OUT/'conclusion_aux_subviajes.json',conclusion)
if r['best_epoch']==0:
 path=OUT/'subviajes_train_s0/features_train.npz';record=read(OUT/'subviajes_train_s0/resultado.json')
 assert path.resolve().is_relative_to(OUT.resolve()) and path.stat().st_size<=600*1024**2
 assert sha(path)==record['outputs'][path.name]['sha256']
 size=path.stat().st_size;path.unlink()
 write(OUT/'subviajes_train_s0/limpieza_cache_descartado.json',dict(deleted_exact_file=path.name,bytes=size,
  reason='Completed auxiliary trial did not improve original reference; keep exact queries/labels/provenance and source.',
  rebuild_script='reconstruir_features_subviajes.py',all_completed_metrics_predictions_preserved=True,
  protected_model_unchanged=sha(BEST/'modelo.pt')==r['config']['parent_sha256']))
write(OUT/'pipeline_subviajes_estado.json',dict(stage='complete',learned_improvement=r['best_epoch']>0,control_run=control is not None,goal='ACTIVE; continue distinct alternatives if needed',Test='sealed'))
print(json.dumps(dict(stage='complete',learned_improvement=conclusion['learned_improvement'],seconds=conclusion['seconds'])),flush=True)
