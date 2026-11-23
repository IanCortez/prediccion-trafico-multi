"""Audit the closed matched geometryprior correction; preserve every qualified checkpoint."""
from base_replanteo import *
run=OUT/'prior_arco_subtrayectos_s0';r=read(run/'resultado.json');assert r['complete'] and r['fixed_epoch']==6
z=dict(np.load(run/'validacion.npz'));truth=np.load(DATA/'validation_y.npy');d=dict(np.load(DATA/'validation_x.npz'))
for key in ['row','timestamp']:assert np.array_equal(z[key],d[key])
assert np.array_equal(z['true'],truth) and len(truth)==127941 and metric(truth,z['pred'])==r['validation']
assert sha(run/'validacion.npz')==r['predictions_sha256']
assert r['all34input_features_normalizer_preserved'] and r['raw65_arc_geometry_exact'] and r['raw_reverse_singleton_exact']
assert read(run/'inicializacion_pareada.json')['all_normalizers_exact']
assert read(run/'inicializacion_pareada.json')['identical_time_control_initial_neural_parameters']
assert len(read(run/'historia.json'))==6 and all(v['primary_train_n']==1023485 for v in read(run/'historia.json'))
for name,digest in r['sources'].items():assert sha(run/'fuentes'/name)==digest
manifest=OUT/'modelos_prior_arco_protegidos.json';models=read(manifest) if manifest.exists() else dict(candidates=[])
if r['qualified']:
 path=Path(r['model_file']).resolve();assert path.is_relative_to(run) and sha(path)==r['model_sha256'] and path.stat().st_size<5*1024**2
 parent=read(OUT/r['config']['parent_id']/'resultado.json');assert sha(parent['model_file'])==r['config']['parent_sha256']
 meta=read(OUT/'indices_subtrayectos_s0/resultado.json')
 deps={'memoria_puntos_Train_s0/puntos_Train.npz':r['config']['source_memory_sha256'],
 'distancia_arco_puntos_Train_s0/distancia_acumulada_m.npz':r['prefix_sha256'],
 'indices_subtrayectos_s0/train_indices.npz':meta['outputs']['train']['sha256'],
 'indices_subtrayectos_s0/validation_indices.npz':meta['outputs']['validation']['sha256'],
 str((run/'constructor_tokens.npz').relative_to(OUT)):r['normalization_sha256'],
 str(Path(parent['model_file']).relative_to(OUT)):parent['model_sha256']}
 item=dict(id=r['id'],validation=r['validation'],model_file=r['model_file'],model_sha256=r['model_sha256'],dependencies=deps,
  checkpoint_kind='Branchonly, protected30componentparent required; original34Dtimefeatures plusauxiliaryarclengthgeometryprior.')
 old=[v for v in models['candidates'] if v['id']==r['id']]
 if old:assert old[0]==item
 else:models['candidates'].append(item)
 write(manifest,models)
proof=run/'reproduccion_independiente.json';replayed=proof.exists() and read(proof)['passed']
conclusion=dict(complete=True,validation=r['validation'],paired_vs_time_sampling=r['paired_vs_time_sampling'],paired_vs_best=r['paired_vs_best'],
 qualified=r['qualified'],useful_gate=r['meaningful_gate_passed'],independent_replay=bool(replayed),semantic_correction_verified_onTrain=True,
 all_original_queries_and_times_preserved=True,original_inputs_normalization_and_initial_neural_parameters_matched=True,
 interpretation='Same6epochcapacity, geometricauxiliarypriorcorrected. ReusedValidation andone trainingseed; no independentconfirmation or universalcausalclaim.',
 next='Reloadqualifiedcheckpoint before anypromotion. No epoch/neighbor/bandwidth sweep.',Test='SEALED',goal='ACTIVE')
write(run/'conclusion.json',conclusion)
p=BASE/'RESUMEN_RESULTADOS.md';s=p.read_text();marker='<!-- resultado_prior_arco_subtrayectos_s0 -->'
if marker not in s:
 s+='\n\n'+marker+'\n## Corrección del prior geométrico de subtrayectos\n\n'
 s+='Se conservaron las 34 variables temporales, normalización, arquitectura, inicialización neuronal y seis épocas. Solo se corrigió la referencia geométrica auxiliar para usar posiciones por distancia de arco. MAE final: '+format(r['validation']['mae_s'],'.6f')+' s sobre los mismos 127.941 viajes. Criterio práctico: '+('cumplido' if r['meaningful_gate_passed'] else 'no cumplido')+'. Todas las predicciones y métricas quedan conservadas; Test cerrado.\n'
 p.write_text(s,encoding='utf-8')
print(json.dumps(conclusion,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v)))
