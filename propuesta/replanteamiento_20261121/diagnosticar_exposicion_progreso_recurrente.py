"""Post-result diagnostics; never substitutes privileged selection for a public OD score."""
from base_replanteo import *
access=deny_test();start=time.time();run=OUT/'progreso_vial_recurrente_s0';bf=run/'etapa_B'
result=read(bf/'resultado.json');assert result['complete']
main=dict(np.load(bf/'validacion.npz'));tr=dict(np.load(DATA/'train_x.npz'));va=dict(np.load(DATA/'validation_x.npz'))
eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index']
manifest=read(run/'resumenes_publicos/resultado.json')
summary={}
for split,d in [('train',tr),('validation',va)]:
 p=run/'resumenes_publicos'/(split+'_resumenes.npz');assert sha(p)==manifest['outputs'][split]['sha256']
 z=dict(np.load(p));assert np.array_equal(z['row'],d['row']) and np.array_equal(z['timestamp'],d['timestamp'])
 summary[split]=z['summary']
 # Omittedattachmentfallback, failed/capped andarrived probabilities must exhaustmass.
 error=np.abs(summary[split][:,5]+summary[split][:,7]+summary[split][:,8]-1)
 assert error.max()<2e-5,(split,float(error.max()))
y=np.load(DATA/'train_y.npy');v=np.load(DATA/'validation_y.npy')
assert np.array_equal(main['true'],v)
diagnostics={}
for split,ids,truth in [('train',eligible,y[eligible]),('validation',np.arange(len(v)),v)]:
 s=summary[split][ids]
 errors={}
 for name,index in [('root_median',1),('public_path_mixture_median',14),('public_path_weighted_mean',3)]:
  pred=s[:,index].astype(np.float64).clip(60,1785)
  errors[name]=dict(**metric(truth,pred),mean_signed_error_s=float(np.mean(pred-truth)))
 spread=s[:,17:21].std(1)
 diagnostics[split]=dict(public_forecasts=errors,
  fallback_probability_mean=float(s[:,5].mean()),tail_probability_mean=float(s[:,7].mean()),
  completed_probability_mean=float(s[:,8].mean()),
  root_fallback_probability_quantiles=np.quantile(s[:,5],[0,.25,.5,.75,1]),
  tail_probability_quantiles=np.quantile(s[:,7],[0,.25,.5,.75,1]),
  style_eta_sd_s_quantiles=np.quantile(spread,[0,.25,.5,.75,.9,.99,1]),
  fraction_style_eta_SD_below1second=float(np.mean(spread<1)),
  fraction_style_eta_SD_below5seconds=float(np.mean(spread<5)),
  prior_entropy_mean=float(s[:,11].mean()),prior_max_mean=float(s[:,10].mean()),
  warning='LowETAspreadcanindicatestyleforecastcollapse butdoesnotproveidenticalroutes. No routearrays/actualqueryroute are used.')
groups={}
def grouped(field,indices):
 out=[]
 for label,mask in indices:
  n=int(mask.sum())
  if n==0:continue
  out.append(dict(group=label,n=n,new=metric(v[mask],main['pred'][mask]),reference=metric(v[mask],main['reference'][mask]),
   delta_mae_s=float(np.mean(np.abs(main['pred'][mask]-v[mask])-np.abs(main['reference'][mask]-v[mask])))))
 groups[field]=out
s=summary['validation']
grouped('predicted_tail_probability',[(name,mask) for name,mask in [
 ('below.1',s[:,7]<.1),('.1to.5',(s[:,7]>=.1)&(s[:,7]<.5)),('atleast.5',s[:,7]>=.5)]])
grouped('omitted_attachment_probability',[(name,mask) for name,mask in [
 ('below.25',s[:,5]<.25),('.25to.5',(s[:,5]>=.25)&(s[:,5]<.5)),('atleast.5',s[:,5]>=.5)]])
lat1=np.radians(va['o_lat']);lat2=np.radians(va['d_lat'])
chord=6371*2*np.arcsin(np.sqrt(np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(np.radians(va['d_lon']-va['o_lon'])/2)**2))
grouped('raw_OD_chord_km',[(str(lo)+'to'+str(hi),(chord>=lo)&(chord<hi)) for lo,hi in [(0,1),(1,2),(2,4),(4,8),(8,np.inf)]])
interval=main['quantiles']
diagnostics['uncertainty']=dict(coverage80=float(np.mean((v>=interval[:,0])&(v<=interval[:,2]))),
 width80_s_mean=float(np.mean(interval[:,2]-interval[:,0])),interpretation='Descriptive reusedValidation only; no posthoccalibration.')
conclusion=dict(complete=True,stage='Post-result diagnostics; no newmodelselection orvalidationtuning',
 goal='ACTIVE',Test='sealed',train_vs_validation=diagnostics,conditional_errors=groups,
 mass_identity_verified_all_queries=True,all_primary_queries_kept=True,
 limits=['Trainforecasts are in-sample for frozenrepresentation; stageB usesin-samplesummaries. Train/Valgap helpsdetectoverfit, not a causalproof.',
 'Teacherrouteaccuracy doesnotmeasure targetfreewholetriprouteaccuracy.',
 'Pseudotimes inherited from15secondGPS allocation; residualdelay is notannotated idling.',
 'Tail/head split andstyle dispersion are modelpredictions; no GTroute/noactualqueryN used.',
 'Feature29 is minoverencoded16public distances, including0forunreachablepairs; interpretasencodedminimum, notalwaysshortestvaliddistance.'],
 predictions_sha256=result['predictions_sha256'],source_sha256=sha(__file__),actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(OUT/'diagnostico_exposicion_progreso_recurrente.json',conclusion)
print(json.dumps(dict(complete=True,diagnostics=diagnostics,groups=groups,seconds=time.time()-start),default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v)),flush=True)
