"""Recover after verified terminal OOM; keep executed sources and epoch1 evidence."""
from base_replanteo import *
import torch,ast,shutil
run=OUT/'mediana_directa_s0';start=time.time()
assert not (run/'resultado.json').exists()
original=OUT/'entrenar_mediana_directa.py'
assert sha(original)==sha(run/'fuentes/entrenar_mediana_directa.py')
history=read(run/'historia.json');assert len(history)==1 and history[0]['epoch']==1
checkpoint=run/'ultima_reanudacion.pt';s=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert s['epoch']==1 and s['history']==history
pred=dict(np.load(run/'predicciones_avance.npz'));reference=dict(np.load(OUT/'historica_fresca_s0/validacion.npz'))
assert np.array_equal(pred['epoch_0'],reference['pred']) and metric(reference['true'],pred['epoch_1'])==history[0]['validation']
archive=run/'recuperacion_memoria_20261122';assert not archive.exists();archive.mkdir()
for name in ['historia.json','predicciones_avance.npz','protocolo.json']:
 shutil.copy2(run/name,archive/name)
shutil.copy2(OUT/'mediana_directa_s0.log',archive/'fallo_original.log')
text=original.read_text(encoding='utf-8-sig')
patches=[
 ("import argparse,torch,shutil","import argparse,torch,shutil,gc"),
 ("sources=['entrenar_mediana_directa.py',","sources=['entrenar_mediana_directa.py','entrenar_mediana_directa_recuperar.py',"),
 ("model.eval();continuous=[];points=[]","gc.collect();torch.cuda.empty_cache();model.eval();continuous=[];points=[]"),
 ("members,_=model.compute(*batch,rp);value=members.mean(1)","members,student_distribution=model.compute(*batch,rp);del student_distribution;value=members.mean(1)"),
 ("opt.zero_grad(set_to_none=True);members,_=model.training_forward(x[ix],c[ix],h[ix],tpoint[ix])","opt.zero_grad(set_to_none=True);members,student_distribution=model.training_forward(x[ix],c[ix],h[ix],tpoint[ix]);del student_distribution")
]
for old,new in patches:
 assert text.count(old)==1,old;text=text.replace(old,new)
ast.parse(text)
recovery=OUT/'entrenar_mediana_directa_recuperar.py';assert not recovery.exists();recovery.write_text(text,encoding='utf-8')
shutil.copy2(recovery,run/'fuentes'/recovery.name)
record=dict(cause='Second evaluation retained previous2048row22component distribution in the underscore variable, increasing peak memory. Process30436 and serial session28304 terminal before this change.',
 completed_epochs_preserved=1,epoch1_validation=history[0]['validation'],
 fix='Separate recovery runner releases unused distributions immediately and clears unused CUDA cache before evaluation. Mathematical model, optimizer, batching, seed,FP32,canonical2048,labels and cohort unchanged.',
 original_executed_trainer_sha256=sha(original),recovery_runner_sha256=sha(recovery),
 unchanged_model_sha256=sha(OUT/'modelo_mediana_directa.py'),resume_checkpoint_epoch1_sha256=sha(checkpoint),
 epoch1_predictions_sha256=sha(archive/'predicciones_avance.npz'),protocol_sha256=sha(run/'protocolo.json'),
 reference_cache_sha256=sha(OUT/'datos_mediana_directa_s0/puntos_referencia.npz'),Test_access=False,
 note='Resume epoch2 from saved epoch1; no replay/selection of a completed epoch. Control, if justified, must use the same recovery runner.')
write(archive/'resultado.json',record)
state=read(OUT/'CONTINUIDAD_OPERATIVA.json');state['current_trial']['stage']='Original30436 terminated OOM during epoch2 evaluation. Epoch1 preserved; separate source-preserving memory-only recovery runner prepared.'
state['current_trial']['pid']=None;state['current_trial']['exec_session_id']=None
state['mediana_memory_recovery']=record;write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
print(json.dumps(dict(epoch1_metrics_preserved=history[0]['validation'],resume_epoch=2,source_preserved=True,memory_only_recovery_prepared=True)))
