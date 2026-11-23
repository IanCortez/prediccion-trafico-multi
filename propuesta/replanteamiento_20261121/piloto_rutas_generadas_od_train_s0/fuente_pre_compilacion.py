"""Actual raw5 OD gate using generated road routes, not held true paths or endpoints."""
from base_replanteo import *
import shutil,math
from numba import set_num_threads
from catboost import CatBoostClassifier,CatBoostRegressor
from consulta_mapa2014 import resources,ORIGIN
from inferencia import feature_input
from rutas_politica_vial import candidate_features,tree_scores,generate_queries
access=deny_test();start=time.time();set_num_threads(2);run=OUT/'piloto_rutas_generadas_od_train_s0';assert not run.exists();run.mkdir();shutil.copy2(__file__,run/'fuente_ejecutada.py');shutil.copy2(OUT/'rutas_politica_vial.py',run/'generador_congelado.py')
policy_run=OUT/'piloto_politica_vial_train_s0';policy_record=read(policy_run/'resultado.json');assert policy_record['passed'] and sha(policy_run/'politica.cbm')==policy_record['model_sha256']
source=OUT/'oraculo_tiempo_calle_train_s0';oracle=read(source/'resultado.json');assert oracle['passed']
plan=dict(scope='Chronological heldTrain OD prediction gate; generated routes frompublic5fields, no heldactualroute/current/endpoint IDs.',
 fit_n=40000,held_n=10000,seed=20261122,control='Exact206public40kfit/10kheld1000treecontrol fromstreetoracle; verify predictions/row/timestamp/labels.',
 route='Learned numeric roadchoicepolicy trained onlyon40kfit actualroutes; generateall16public4x4endpointpairs. Same modelandclockfeatures as predeclaredteacherstatediagnostic.',
 termination='Absorbingarrival atpublicdestinationpartialoffset once; fixed192chaincap andmaximum2visits then publicshortestpathtail. CapindependentofobservedN. Deadend/invalidroutes masked without removingquery.',
 features='Projection-priorweightedmean/std90generatedroutegeometrydescriptors plusvalidity/fallbackfractions andprojectionentropy. Addto206publiccontrol.',
 time_model=read(source/'plan.json')['model'],gate='At least2seconds heldTrainMAE improvement andupperdaybootstrapdeltaCI<0. Otherwise do not expand fullprimaryroutebank.',
 numeric_policy='Exportexistingclassifier numeric symmetric trees; rawlogit equivalence toCatBoost verifiedbefore generation.',
 resources='CPU2; routesummarystreamchunks256queries; artifacts<=200MiB. NoGPU, nofulltrajectorycache.',Validation=False,Test=False)
write(run/'plan.json',plan)
model=CatBoostClassifier();model.load_model(str(policy_run/'politica.cbm'));model.save_model(str(run/'politica_numerica.json'),format='json');export=read(run/'politica_numerica.json')
trees=export['oblivious_trees'];depth=max(len(t['splits']) for t in trees);split_feature=np.full((len(trees),depth),-1,np.int32);border=np.zeros((len(trees),depth),np.float64);leaf=np.zeros((len(trees),2**depth),np.float64)
for i,t in enumerate(trees):
 for j,s in enumerate(t['splits']):assert s['split_type']=='FloatFeature';split_feature[i,j]=s['float_feature_index'];border[i,j]=s['border']
 leaf[i,:len(t['leaf_values'])]=t['leaf_values']
scale=float(export['scale_and_bias'][0]);bias=float(export['scale_and_bias'][1][0])
raw,c,sample,tree,routes,profiles=resources();set_num_threads(2);D,conn,routing=routes['length'];g=dict(np.load(OUT/'banco_valor_vial_s0/grafo_estatico.npz'))
sx=raw['xy'][c['junction_raw'][c['chain_src']]];dx=raw['xy'][c['junction_raw'][c['chain_dst']]];vec=dx-sx;heading=vec/np.maximum(np.linalg.norm(vec,axis=1,keepdims=True),.001)
angle=(np.arctan2(vec[:,1],vec[:,0])+2*np.pi)%(2*np.pi);orientation=np.floor(angle*8/(2*np.pi)).astype(np.int32)%8
speed=c['chain_length']/np.maximum(c['chain_free'],.001);speedbin=np.digitize(speed,[4,7,10,14,20]).astype(np.int32)
flat=c['flat_incoming_arc'];valid=flat>=0;pc=np.repeat(np.arange(len(c['chain_src']),dtype=np.int32),np.diff(c['chain_offsets']));edge_chain=np.full(len(raw['u']),-1,np.int32);edge_chain[flat[valid]]=pc[valid]
with np.load(source/'auxiliares_Train.npz') as z:q={key:z[key] for key in ['row','train_index','timestamp','true']}
d,X,names=feature_input('train',read(BEST/'resultado.json'));keep=np.asarray([not n.startswith('mm_') for n in names]);assert keep.sum()==206
ix=q['train_index'];assert np.array_equal(d['row'][ix],q['row']);public=X[ix][:,keep];del X
with np.load(OUT/'banco_valor_vial_s0/train_proyecciones.npz') as z:bank={key:z[key][ix] for key in ['chain','fraction','error_m','reachable']}
ol=np.column_stack([d['o_lon'][ix],d['o_lat'][ix]]);dl=np.column_stack([d['d_lon'][ix],d['d_lat'][ix]])
chords=np.linalg.norm(np.column_stack([(dl[:,0]-ol[:,0])*111195*np.cos(np.deg2rad(ORIGIN[1])),(dl[:,1]-ol[:,1])*111195]),axis=1)
departures=(q['timestamp']%86400)/86400*2*np.pi
# Real finite mapstatefeatures; no dummy-model numerical proof.
cur=int(bank['chain'][0,0,0]);dest=int(bank['chain'][0,1,0]);candidates=g['neighbors'][cur][g['neighbor_valid'][cur]]
F,cost=candidate_features(cur,dest,cur,candidates,g['static'],sx,dx,heading,c['chain_length'],D,c['chain_dst'],c['chain_src'],departures[0])
native=model.predict(F,prediction_type='RawFormulaVal');ours=tree_scores(F,split_feature,border,leaf,scale,bias);error=float(np.max(np.abs(native-ours)));assert error<1e-8
args=(g['static'],sx,dx,heading,c['chain_length'],c['chain_free'],orientation,speedbin,g['neighbors'],g['neighbor_valid'],
 D,conn,routing,raw['u'],raw['v'],raw['length'],c['chain_src'],c['chain_dst'],c['chain_offsets'],c['prefix_length'],c['flat_incoming_arc'],edge_chain,split_feature,border,leaf,scale,bias,192)
chains=bank['chain'].astype(np.int32);fractions=bank['fraction'].astype(np.float32)
test,fail,tail=generate_queries(chains[:3],fractions[:3],bank['reachable'][:3],chords[:3],departures[:3],*args)
reverse,rf,rt=generate_queries(chains[:3][::-1],fractions[:3][::-1],bank['reachable'][:3][::-1],chords[:3][::-1],departures[:3][::-1],*args)
singleton,sf,st=generate_queries(chains[:1],fractions[:1],bank['reachable'][:1],chords[:1],departures[:1],*args)
assert np.array_equal(test,reverse[::-1]) and np.array_equal(test[:1],singleton) and np.array_equal(fail,rf[::-1]) and np.array_equal(tail,rt[::-1])
write(run/'verificacion_generador.json',dict(passed=True,raw_policy_score_max_error=error,reverse_singleton_descriptor_exact=True,
 no_observed_N_route_or_time_in_generator_API=True,absorbingarrival_and_publictail=True))
n=len(ix);summary=np.empty((n,183),np.float32);bad_total=0;tail_total=0;routes_total=0
for a in range(0,n,256):
 b=min(a+256,n);values,failed,fallback=generate_queries(chains[a:b],fractions[a:b],bank['reachable'][a:b],chords[a:b],departures[a:b],*args)
 eo=bank['error_m'][a:b,0].astype(np.float64);ed=bank['error_m'][a:b,1].astype(np.float64)
 logits=(-.5*((eo[:,:,None]/30)**2+(ed[:,None,:]/30)**2)).reshape(-1,16);logits=np.where(~failed,logits,-1e9);weights=np.exp(logits-logits.max(1,keepdims=True))*(~failed);weights/=np.maximum(weights.sum(1,keepdims=True),1e-20)
 mean=np.einsum('bk,bkf->bf',weights,values);var=np.einsum('bk,bkf->bf',weights,(values-mean[:,None])**2);entropy=-(weights*np.log(np.maximum(weights,1e-20))).sum(1)/np.log(16)
 summary[a:b]=np.column_stack([mean,np.sqrt(np.maximum(var,0)),(~failed).mean(1),fallback.mean(1),entropy])
 bad_total+=int(failed.sum());tail_total+=int(fallback.sum());routes_total+=failed.size
 if (b%4096==0) or b==n:print(json.dumps(dict(stage='raw5_generated_routes',queries=b,total=n,failed_fraction=bad_total/routes_total,fallback_fraction=tail_total/routes_total,seconds=time.time()-start)),flush=True)
np.savez_compressed(run/'descriptores_generados_Train.npz',row=q['row'],timestamp=q['timestamp'],features=summary)
assert np.isfinite(summary).all()
p0=dict(np.load(source/'predicciones_held_Train.npz'));nf=plan['fit_n'];y=q['true'].astype(np.float64)
for key in ['row','timestamp']:assert np.array_equal(p0[key],q[key][nf:])
assert np.array_equal(p0['true'],y[nf:]) and metric(y[nf:],p0['control'])==oracle['metrics']['control']
print(json.dumps(dict(stage='fitting_generated_route_time',seconds=time.time()-start)),flush=True)
aug=np.concatenate([public,summary],1);tte=CatBoostRegressor(**plan['time_model']);tte.fit(aug[:nf],y[:nf]);pred=tte.predict(aug[nf:]);comparison=paired(y[nf:],pred,p0['control'],q['timestamp'][nf:]);comparison['interpretation']='ChronologicalheldTrain raw5ODgate, noheldactualroute input; not primaryValidation.'
passed=bool(comparison['delta_mae_s']<=-2 and comparison['ci95_s'][1]<0)
if passed:tte.save_model(str(run/'tiempo_rutas_generadas.cbm'))
np.savez_compressed(run/'predicciones_held_Train.npz',row=q['row'][nf:],timestamp=q['timestamp'][nf:],true=y[nf:],pred=pred,control=p0['control'])
result=dict(passed=passed,scope=plan['scope'],metrics=metric(y[nf:],pred),control=metric(y[nf:],p0['control']),comparison=comparison,
 generator_scores_verified=True,all16publicendpointpairs=True,failed_route_fraction=bad_total/routes_total,shortest_tail_fraction=tail_total/routes_total,
 source_sha256=sha(run/'fuente_ejecutada.py'),generator_sha256=sha(run/'generador_congelado.py'),policy_sha256=policy_record['model_sha256'],
 predictions_sha256=sha(run/'predicciones_held_Train.npz'),features_sha256=sha(run/'descriptores_generados_Train.npz'),
 no_held_trajectory_or_actual_HMMendpoints_or_observed_N_used=True,Validation_access=False,Test_access=False,seconds=time.time()-start)
assert sum(p.stat().st_size for p in run.iterdir() if p.is_file())<200*1024**2
write(run/'resultado.json',result);print(json.dumps(dict(passed=passed,metrics=result['metrics'],control=result['control'],delta_mae_s=comparison['delta_mae_s'],seconds=result['seconds'])),flush=True)
