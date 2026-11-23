"""Freeze the matched source-value control before the treatment sees Validation."""
from base_replanteo import *
import ast
treatment=OUT/'subtrayectos_conjuntos_s0'
assert not (treatment/'resultado.json').exists()
source=treatment/'fuentes/entrenar_subtrayectos_conjuntos.py'
assert sha(source)==sha(OUT/'entrenar_subtrayectos_conjuntos.py')
s=source.read_text()
s=s.replace("previous=read(OUT/'atencion_subtrayectos_s0/resultado.json');assert previous['complete'] and not previous['meaningful_gate_passed']",
 """treatment=read(OUT/'subtrayectos_conjuntos_s0/resultado.json')
assert treatment['complete'] and treatment['meaningful_gate_passed'],'Controlnotauthorizedwithoutpredeclaredusefulgain'
previous=read(OUT/'atencion_subtrayectos_s0/resultado.json');assert previous['complete']""")
s=s.replace("reference=read(OUT/'nuevo_mejor_verificado.json');parent_run=OUT/reference['id']",
 """reference=read(OUT/treatment['config']['comparison_reference_id']/'resultado.json')
assert reference['model_sha256']==treatment['config']['comparison_reference_sha256']
parent_run=OUT/reference['id']""")
s=s.replace("run=OUT/'subtrayectos_conjuntos_s0'","run=OUT/'control_subtrayectos_conjuntos_s0'")
s=s.replace("'entrenar_subtrayectos_conjuntos.py'","'entrenar_control_subtrayectos_conjuntos.py'")
s=s.replace("config=dict(epochs=24,","config=dict(mode='matched_source_value_control',treatment_id='subtrayectos_conjuntos_s0',treatment_result_sha256=sha(OUT/'subtrayectos_conjuntos_s0/resultado.json'),epochs=24,")
s=s.replace("write(run/'inicializacion_fresca.json',dict(initial_state_sha256=initial.hexdigest()",
 "assert initial.hexdigest()==read(OUT/'subtrayectos_conjuntos_s0/inicializacion_fresca.json')['initial_state_sha256']\nwrite(run/'inicializacion_fresca.json',dict(initial_state_sha256=initial.hexdigest()")
s=s.replace("def tokens(subquery,indexbank,index):",
 """def remove_source_values(raw):
 # Keep source availability, endpoints, localclock, start/stopfractions and rankingmetadata.
 # Remove measuredgeometry/path/sourceelapsedtime AND full-parentlogduration (column32),
 # otherwise fullParentT times intervalfraction would reconstructsourceelapsedtime.
 raw=raw.clone();raw[...,:17]=model.token_mean[:17].to(raw.dtype);raw[...,32]=model.token_mean[32].to(raw.dtype)
 return raw
def tokens(subquery,indexbank,index):""")
s=s.replace(" return sub\n"," return remove_source_values(sub)\n")
s=s.replace("raw=library.features(st['xy_m'],st['hour'],st['weekend'],source[0],source[1],source[2])",
 "raw=remove_source_values(library.features(st['xy_m'],st['hour'],st['weekend'],source[0],source[1],source[2]))")
assert 'full-parentlogduration' in s and s.count('remove_source_values(')==3
path=OUT/'entrenar_control_subtrayectos_conjuntos.py';assert not path.exists();ast.parse(s);path.write_text(s)
write(OUT/'plan_control_subtrayectos_conjuntos.json',dict(
 frozen_before_treatment_Validation=True,treatment_source_sha256=sha(source),control_source_sha256=sha(path),
 trigger='Treatment at least.5sbetterthan129.990034andupperexploratorydayCI<0. Otherwise do notrun.',
 matched='ExactsamefreshinitialtensorSHA, originalTrainqueries, sourceindices, optimizer, batches, loss, seed and24fixedepochs.',
 intervention='Replace sourcegeometry/path/time columns0..16 and full-parentlogtime32 withTrainmeans. Keep availability, endpointdelta, clock/retrievalprior, andstart/stopfractions.',
 no_duration_reconstruction='FullparentT removedaswell asintervalT; keeping fullparentT andintervalfraction wouldinvalidate thecontrol.',
 limitation='Remaining context proxies may be informative. Control measures incremental measuredsourcevalues inthisarchitecture, not universalcausality.',
 Test='SEALED',primary_cohort_unchanged=True))
print('Matchedcontrol frozen before treatmentValidation; not launched.')
