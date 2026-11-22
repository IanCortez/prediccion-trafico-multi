from base_replanteo import *
import ast
source=OUT/'continuar_tras_prefijo_nativo.py'
s=source.read_text(encoding='utf-8-sig')
assert 'for pid in [20612,33120]:' in s
s=s.replace('for pid in [20612,33120]:','for pid in [35928]:')
s=s.replace('original_processes=[20612,33120]','original_processes=[35928]')
target=OUT/'continuar_tras_prefijo_acelerado.py'
assert not target.exists()
ast.parse(s);target.write_text(s,encoding='utf-8')
active=dict(experiment='prefijo_nativo_s0/evaluacion_acelerada',pid=35928,session=46924,device='GPU',max_GPU_jobs=1,training_finished=True,epochs=10,steps=79960,checkpoint_sha256=read(OUT/'prefijo_nativo_s0/congelacion_para_inferencia.json')['model_sha256'],do_not_restart_training=True)
for name in ['sesiones_activas.json','sesiones_extra.json']:write(OUT/name,dict(active=[active],goal='active',Test='sealed'))
c=read(OUT/'CONTINUIDAD_OPERATIVA.json')
c['current_trial']=active
c['active_unified_session']=46924
c['superseded_sessions']=[9112,46174]
c['do_not_wait_superseded_numeric_pids']=[20612,33120,35436]
c['replacement_reason']='10epochs79960steps completed and checkpoint frozen. Original16query inference was slow. Replaced inference only, source and weights unchanged; new batch chosen without MAE and must match512point probe.'
c['next_pipeline']='continuar_tras_prefijo_acelerado.py; waits on actual replacement35928handle, then full-result conclusion and conditional CPU gate'
write(OUT/'CONTINUIDAD_OPERATIVA.json',c)
print('Continuidad conectada a la evaluación nueva; entrenamiento finalizado e intacto.')
