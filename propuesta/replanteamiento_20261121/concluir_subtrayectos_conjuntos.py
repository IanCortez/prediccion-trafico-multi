"""Audit closed full-primary joint/control result and register every qualified full model."""
from base_replanteo import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--run',default='subtrayectos_conjuntos_s0');a=p.parse_args()
run=(OUT/a.run).resolve();assert run.parent==OUT.resolve()
r=read(run/'resultado.json');assert r['complete'] and r['fixed_epoch']==24
z=dict(np.load(run/'validacion.npz'));d=dict(np.load(DATA/'validation_x.npz'));truth=np.load(DATA/'validation_y.npy')
for key in ['row','timestamp']:assert np.array_equal(z[key],d[key])
assert np.array_equal(z['true'],truth) and len(truth)==127941 and metric(truth,z['pred'])==r['validation']
assert np.array_equal(z['pred'],z['quantiles'][:,1])
assert sha(run/'validacion.npz')==r['predictions_sha256']
for name,digest in r['sources'].items():assert sha(run/'fuentes'/name)==digest
assert r['raw65_cached_sources_tokens_full_quantiles_exact'] and r['raw_reverse_singleton_exact']
history=read(run/'historia.json');assert len(history)==24 and all(item['primary_train_n']==1023485 for item in history)
manifest=OUT/'modelos_subtrayectos_conjuntos_protegidos.json'
models=read(manifest) if manifest.exists() else dict(candidates=[])
if r['qualified']:
 path=Path(r['model_file']).resolve();assert path.is_relative_to(run) and sha(path)==r['model_sha256']
 assert path.stat().st_size<100*1024**2,'Unexpectedlargecheckpoint; inspectbuffers'
 meta=read(OUT/'indices_subtrayectos_s0/resultado.json')
 dependencies={'memoria_puntos_Train_s0/puntos_Train.npz':r['config']['source_memory_sha256'],
 'indices_subtrayectos_s0/train_indices.npz':meta['outputs']['train']['sha256'],
 'indices_subtrayectos_s0/validation_indices.npz':meta['outputs']['validation']['sha256'],
 str((run/'constructor_tokens.npz').relative_to(OUT)):r['normalization_sha256']}
 item=dict(id=r['id'],validation=r['validation'],model_file=r['model_file'],model_sha256=r['model_sha256'],
  dependencies=dependencies,checkpoint_kind='Fulljointmodel, no neuralparent checkpoint required.',qualified=True)
 old=[v for v in models['candidates'] if v['id']==r['id']]
 if old:assert old[0]==item
 else:models['candidates'].append(item)
 write(manifest,models)
controls=None
if r['id']=='control_subtrayectos_conjuntos_s0':
 treatment=read(OUT/'subtrayectos_conjuntos_s0/resultado.json')
 tz=dict(np.load(OUT/'subtrayectos_conjuntos_s0/validacion.npz'))
 for key in ['row','timestamp','true']:assert np.array_equal(z[key],tz[key])
 assert read(run/'inicializacion_fresca.json')['initial_state_sha256']==read(OUT/'subtrayectos_conjuntos_s0/inicializacion_fresca.json')['initial_state_sha256']
 controls=paired(truth,tz['pred'],z['pred'],d['timestamp'])
 controls['interpretation']='Treatmentminusmatchedsourcevaluescontrol; fixed24epochs, samefreshinitialtensorSHA. ReusedValidation exploratory.'
replayfile=run/'reproduccion_independiente.json'
replayed=replayfile.exists() and read(replayfile)['passed']
conclusion=dict(complete=True,id=r['id'],validation=r['validation'],paired_vs_frozen_reference=r['paired_vs_parent'],
 useful_gain_gate=r['meaningful_gate_passed'],qualified_model_preserved=r['qualified'],full_checkpoint_independently_replayed=bool(replayed),
 all24epochs_all_originalTrainqueries=True,all127941primary_queries_and_labels_verified=True,
 raw65_source_token_quantile_order_singleton=True,matched_source_value_comparison=controls,
 next='Runindependentreloadifqualified; runmatchedcontrolonlyifpreregisteredusefulgatepassed. No newepochs or neighbors selected.',
 no_confirmation_claim=True,Test='SEALED',goal='ACTIVE',source_sha256=sha(__file__))
write(run/'conclusion.json',conclusion)
summary=BASE/'RESUMEN_RESULTADOS.md';text=summary.read_text();marker='<!-- resultado_'+r['id']+' -->'
if marker not in text:
 text+='\n\n'+marker+'\n## '+r['id']+'\n\n'
 text+='Estado final fijo de 24 épocas; 1.023.485 consultas Train por época y los mismos 127.941 viajes de Validation. MAE '+format(r['validation']['mae_s'],'.6f')+' s; RMSE '+format(r['validation']['rmse_s'],'.6f')+' s. '
 text+='Criterio de mejora útil: '+('cumplido' if r['meaningful_gate_passed'] else 'no cumplido')+'. '
 text+='Todos los tiempos y consultas originales se conservaron. Métricas, predicciones y fuentes completas guardadas; pesos '+('protegidos por umbral o mínimo' if r['qualified'] else 'descartados por no cumplir el umbral')+'. Test cerrado.\n'
 summary.write_text(text,encoding='utf-8')
print(json.dumps(dict(id=r['id'],validation=r['validation'],qualified=r['qualified'],useful_gate=r['meaningful_gate_passed'],replayed=replayed),default=str))
