"""Derive a separate trainer without modifying the frozen queued request study."""
from base_replanteo import *
src=OUT/'entrenar_aux_solicitud.py'
assert sha(src)==read(OUT/'plan_aux_solicitud.json')['sources'][src.name]
s=src.read_text(encoding='utf-8-sig')
def replace_once(old,new):
 global s
 assert s.count(old)==1,(old,s.count(old))
 s=s.replace(old,new,1)
replace_once('from modelo_aux_solicitud import RequestAuxiliary','from modelo_aux_solicitud import RequestAuxiliary\nfrom perdida_maestro_probabilistico import expected_teacher_score')
replace_once("choices=['control','semantic']","choices=['distill']")
replace_once("weight=.2 if a.arm=='semantic' else 0.;run=OUT/('aux_solicitud_'+a.arm+'_s0_v2')","weight=0.;teacher_weight=.2;run=OUT/'alumno_maestro_s0_v2'")
replace_once("sources=['entrenar_aux_solicitud.py','modelo_aux_solicitud.py','base_replanteo.py']","""teacher_run=OUT/'maestro_cruzado_s0'
teacher_record=read(teacher_run/'resultado.json')
assert teacher_record['all_eligible_rows_assigned_once'] and teacher_record['own_whole_fold_excluded_for_every_row'] and teacher_record['Train_n']==1023485
teacher_file=teacher_run/'parametros_train_fuera_bloque.npz'
assert sha(teacher_file)==teacher_record['parameters_sha256']
teacher_proof=read(OUT/'verificacion_perdida_maestro.json');assert teacher_proof['passed']
assert sha(OUT/'perdida_maestro_probabilistico.py')==teacher_proof['sources']['perdida_maestro_probabilistico.py']
control_record=read(OUT/'aux_solicitud_control_s0_v2/resultado.json')
for key in ['epochs','batch','seed','duration_lr','request_lr','weight_decay','geometry_weight','rps_weight','canonical_inference_batch','precision','parent_sha256','query','primary_train_rule','training_determinism','compatibility_fix']:
 assert config[key]==control_record['config'][key],('Unmatched control',key)
assert control_record['config']['weight']==0
config.update(teacher_weight=teacher_weight,teacher_parameters_sha256=sha(teacher_file),teacher_parameter_file=str(teacher_file.relative_to(ROOT)),soft_target_probability_sha256=sha(OUT/'perdida_maestro_probabilistico.py'),control_source_sha256=sha(OUT/'entrenar_aux_solicitud.py'),matched_control='aux_solicitud_control_s0_v2',query_uses_teacher=False)
sources=['entrenar_alumno_maestro.py','modelo_aux_solicitud.py','base_replanteo.py','perdida_maestro_probabilistico.py']""")
replace_once("y=np.load(DATA/'train_y.npy');v=np.load(DATA/'validation_y.npy');assert len(v)==127941","""y=np.load(DATA/'train_y.npy');v=np.load(DATA/'validation_y.npy');assert len(v)==127941
soft=dict(np.load(teacher_file))
assert np.array_equal(soft['trip_train_index'],eligible) and np.array_equal(soft['row'],tr['row'][eligible])
assert np.array_equal(soft['timestamp'],tr['timestamp'][eligible])
assert np.array_equal(soft['fold'],((tr['timestamp'][eligible]//(7*86400))%5).astype(np.uint8))
mu_array=np.full(len(y),np.nan,np.float32);ls_array=np.full(len(y),np.nan,np.float32)
mu_array[eligible]=soft['mu_log_seconds'];ls_array[eligible]=soft['log_sigma']
assert np.isfinite(mu_array[eligible]).all() and np.isfinite(ls_array[eligible]).all()""")
replace_once("del X,V,gt,geometry,calls","tmu=torch.tensor(mu_array,device='cuda');tls=torch.tensor(ls_array,device='cuda')\ndel X,V,gt,geometry,calls,soft,mu_array,ls_array")
replace_once("return nll+5*rps+.2*geom+weight*request,request","soft_score=expected_teacher_score(out[0],tmu[index],tls[index])\n return nll+5*rps+.2*geom+weight*request+teacher_weight*soft_score,soft_score")
replace_once("request_conditional_NLL=aux/count","teacher_expected_score=aux/count")
compile(s,str(OUT/'entrenar_alumno_maestro.py'),'exec')
(OUT/'entrenar_alumno_maestro.py').write_text(s,encoding='utf-8')
assert sha(src)==read(OUT/'plan_aux_solicitud.json')['sources'][src.name]
# Verify that shared objective and sampling source blocks are literally unchanged.
primary=lambda text:text.split('def objective(index):',1)[1].split(' request=model.request_score',1)[0]
sampling=lambda text:text.split("for selected in ids[torch.randperm",1)[1].split(" prediction=predict()",1)[0]
old=src.read_text(encoding='utf-8-sig')
assert primary(s)==primary(old) and sampling(s)==sampling(old)
plan=dict(hypothesis='Out-of-fit privileged probabilistic teacher gives a softer Train time target while the student query remains OD only.',
 teacher='Five whole-week-fold-excluded CatBoost lognormal teachers using actual geometry only in Train. Two raw parameters per row, not a CDF bank.',
 objective='Original hard NLL +5 hard RPS +0.2 conditional geometry NLL +0.2*(teacher expected NLL+5 soft RPS). Teacher-only RPS variance constant omitted without changing gradient.',
 primary_labels_and_filters='unchanged',query_teacher_input=False,
 matched_control='Reuse exactly predeclared aux_solicitud_control_s0_v2. Same RequestAuxiliary wrapper with zero request loss, same seed, optimizer, batches,4epochs, and literal primary/sampling source.',
 common_primary_source_sha256=hashlib.sha256(primary(s).encode()).hexdigest(),
 common_sampling_source_sha256=hashlib.sha256(sampling(s).encode()).hexdigest(),
 control_source_sha256=sha(src),four_epochs_fixed_before_control_results=True,
 probability_CPU_gate='verificacion_perdida_maestro.json: exact expected-hard-score value/gradient identity and real-Train finite gradients passed',
 GPU_gate='Mandatory finite gradients plus all protected reference predictions exact before any update.',
 execution_order='Wait for actual request-study supervisor and teacher CPU process exits; require both results. No extra GPU simultaneous. A73.5Validation candidate takes priority for freeze/confirmation.',
 limitations='Privileged teacher gain alone does not establish student improvement. Adaptive reused Validation remains exploratory.',
 sources={f:sha(OUT/f) for f in ['entrenar_alumno_maestro.py','perdida_maestro_probabilistico.py','modelo_aux_solicitud.py']},
 goal='active',Test='sealed')
write(OUT/'plan_alumno_maestro.json',plan)
print(json.dumps(dict(prepared=True,control_source_unchanged=True,primary_and_sampling_source_identical=True,sources=plan['sources'])))
