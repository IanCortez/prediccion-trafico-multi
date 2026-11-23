"""One frozen full-primary evaluation of implicit historical subtrajectory memory."""
from base_replanteo import *
import torch,shutil,gc
from inferencia import feature_input
from inferencia_atencion_historia import load_frozen_attention,raw_inputs,iter_attention
access=deny_test();start=time.time()
torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.5)
# Load protected frozen hierarchy before source helper modules.
model,ck,r,parent_run,proto,library,helper,token=load_frozen_attention('atencion_historia_individual_s0')
from recuperar_subtrayectos import SubtrajectoryMemory
from experto_subtrayectos_cdf import combine,quantiles
run=OUT/'memoria_subtrayectos_cdf_s0';run.mkdir(exist_ok=False);(run/'fuentes').mkdir()
sources=['evaluar_memoria_subtrayectos_cdf.py','recuperar_subtrayectos.py','experto_subtrayectos_cdf.py',
         'datos_progreso_recurrente.py','base_replanteo.py','inferencia_atencion_historia.py']
for name in sources:shutil.copy2(OUT/name,run/'fuentes'/name)
proof=read(OUT/'verificacion_memoria_subtrayectos.json');assert proof['passed']
assert proof['model_source_sha256']==sha(OUT/'recuperar_subtrayectos.py')
plan=dict(public_query=['o_lon','o_lat','d_lon','d_lat','timestamp'],primary_Validation=127941,
 original_labels_unchanged=True,Test='sealed',nearest_points_per_endpoint=2048,max_distinct_source_parents=64,
 source_time='15*(j-i) with j-i>=4, sourcechord>=200m, sourceODspeed<=120kmh; public chordratio clipped .5..2',
 source_weights='Spatial sigma50m, Lisbon localclock sigma4h, weekendmismatch multiplier.35',
 source_logtime_sigma=.20,main='evidence_mixture',
 gate='ESS/(ESS+8) times exp(-.5*weightedendpoint_error_squared/50^2)',
 control='kernel_backoff: source density if any, otherwise protectedCDF',
 no_Validation_tuning=True,meaningful_gate='Atleast.5seconds better than protected and upperdayCI<0',
 sources={name:sha(run/'fuentes'/name) for name in sources},reference=r['validation'],
 reference_checkpoint_sha256=r['model_sha256'])
write(run/'protocolo.json',plan)
d,V,names=feature_input('validation',read(BEST/'resultado.json'));assert names==ck['features']
truth=np.load(DATA/'validation_y.npy');saved=dict(np.load(parent_run/'validacion.npz'))
for key in ['row','timestamp']:assert np.array_equal(d[key],saved[key])
assert np.array_equal(truth,saved['true']) and len(truth)==127941
C=np.load(OUT/'banco_proyeccion_incierta_s0/validation_banco.npz')['bank']
H=np.load(OUT/'datos_prototipos_historicos_s0/validation_historicos.npz')['history']
qz=dict(np.load(OUT/'datos_historia_individual_s0/validation_indices.npz'))
for key in ['row','timestamp']:assert np.array_equal(d[key],qz[key])
query={key:qz[key] for key in ['xy_km','hour','weekend','indices']}
bank=SubtrajectoryMemory();bank.build_tree(5)
fields=plan['public_query'];predictions={k:[] for k in ['kernel_backoff','evidence_mixture']};summaries=[]
for first,n,out in iter_attention(model,library,V,C,H,query):
 base=out[0][:n].exp().mean(1).cumsum(-1).cpu().numpy()
 baseq=model.duration.quantiles(out[0])[:n].cpu().numpy()
 assert np.array_equal(baseq[:,1],saved['pred'][first:first+n])
 # Compute parent CDF on its original device before transfer: numpy serial Float32 cumsum differed at threshold; no target tuning.
 assert np.array_equal(quantiles(base),baseq)
 batchparts={k:[] for k in predictions};batchsummary=[]
 for offset in range(0,n,128):
  a=first+offset;b=min(first+n,a+128)
  public={key:d[key][a:b] for key in fields}
  source=bank.query(public);p,s=combine(base[offset:offset+b-a],source)
  for key in batchparts:batchparts[key].append(p[key])
  batchsummary.append(s)
 for key in predictions:predictions[key].append(np.concatenate(batchparts[key]))
 summaries.append(np.concatenate(batchsummary));del out,base
 if first%16384==0:print(json.dumps(dict(completed=first+n,total=len(truth),seconds=time.time()-start)),flush=True)
predictions={key:np.concatenate(v) for key,v in predictions.items()};summary=np.concatenate(summaries)
# Raw-fivefield source, feature, and full predictor reproducibility, without labels.
ii=np.sort(np.random.default_rng(20261122).choice(len(truth),65,replace=False))
items=[{k:int(d[k][i]) if k=='timestamp' else float(d[k][i]) for k in fields} for i in ii]
def predict_raw(items):
 X,RC,RH,RQ,rnames=raw_inputs(items,ck['base_config'],proto,library,helper,token);assert rnames==names
 base=[]
 for f,n,out in iter_attention(model,library,X,RC,RH,RQ):base.append(out[0][:n].exp().mean(1).cumsum(-1).cpu().numpy())
 dd={key:np.array([item[key] for item in items]) for key in fields}
 return combine(np.concatenate(base),bank.query(dd))
raw,rs=predict_raw(items);rev,vs=predict_raw(items[::-1]);single,ss=predict_raw(items[:1])
assert np.array_equal(rs,summary[ii]) and np.array_equal(rs,vs[::-1]) and np.array_equal(rs[:1],ss)
for key in predictions:
 assert np.array_equal(raw[key],predictions[key][ii])
 assert np.array_equal(raw[key],rev[key][::-1]) and np.array_equal(raw[key][:1],single[key])
np.savez_compressed(run/'validacion.npz',row=d['row'],timestamp=d['timestamp'],true=truth,
 pred=predictions['evidence_mixture'][:,1],quantiles=predictions['evidence_mixture'],
 kernel_backoff=predictions['kernel_backoff'][:,1],kernel_quantiles=predictions['kernel_backoff'],
 reference=saved['pred'],source_summary=summary)
metrics={key:metric(truth,p[:,1]) for key,p in predictions.items()}
comparisons={key:paired(truth,p[:,1],saved['pred'],d['timestamp']) for key,p in predictions.items()}
grouped=[]
for lo,hi in [(0,1),(1,4),(4,8),(8,100)]:
 mask=(summary[:,0]>=lo)&(summary[:,0]<hi)
 if mask.any():grouped.append(dict(source_parent_range=[lo,hi],n=int(mask.sum()),
  candidate=metric(truth[mask],predictions['evidence_mixture'][mask,1]),reference=metric(truth[mask],saved['pred'][mask])))
main=comparisons['evidence_mixture'];gate=main['delta_mae_s']<=-.5 and main['ci95_s'][1]<0
result=dict(complete=True,id=run.name,validation=metrics['evidence_mixture'],metrics=metrics,paired_vs_reference=comparisons,
 meaningful_gate_passed=bool(gate),new_numerical_minimum=metrics['evidence_mixture']['mae_s']<r['validation']['mae_s'],
 nonparametric_model=True,required_sources=['memoria_puntos_Train_s0/puntos_Train.npz',r['id']],
 source_bank_sha256=read(OUT/'memoria_puntos_Train_s0/resultado.json')['outputs_sha256'],
 source_summary_columns=['parent_count','effective_parent_count','weighted_endpoint_RMS_m','gate','full_trip_weight','weighted_time_s'],
 source_parent_count_quantiles=np.quantile(summary[:,0],[0,.1,.25,.5,.75,.9,1]),
 fraction_no_source=float(np.mean(summary[:,0]==0)),mean_gate=float(summary[:,3].mean()),coverage_groups=grouped,
 raw65_full_quantiles_exact=True,reverse_singleton_exact=True,all_protected_reference_points_exact=True,
 uncertainty={key:dict(coverage80=float(np.mean((truth>=p[:,0])&(truth<=p[:,2]))),width80_s=float((p[:,2]-p[:,0]).mean())) for key,p in predictions.items()},
 no_query_private_fields=True,Test_access=False,sources=plan['sources'],
 predictions_sha256=sha(run/'validacion.npz'),actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(run/'resultado.json',result);write(run/'consultas_raw_sin_etiquetas.json',items)
print(json.dumps(read(run/'resultado.json')),flush=True)
