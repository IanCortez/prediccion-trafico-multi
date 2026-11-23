"""Full-primary descriptive error audit, never queryroute decoding or modelselection."""
from base_replanteo import *
access=deny_test();start=time.time();run=OUT/'atencion_subtrayectos_s0';r=read(run/'resultado.json')
new=dict(np.load(run/'validacion.npz'));old=dict(np.load(OUT/r['config']['parent_id']/'validacion.npz'))
kernel=dict(np.load(OUT/'memoria_subtrayectos_cdf_s0/validacion.npz'))
d=dict(np.load(DATA/'validation_x.npz'));truth=np.load(DATA/'validation_y.npy')
for key in ['row','timestamp']:
 for z in [new,old,kernel]:assert np.array_equal(z[key],d[key])
for z in [new,old,kernel]:assert np.array_equal(z['true'],truth)
source=kernel['source_summary'];counts=source[:,0];no_source=counts==0
assert np.all(new['pred'][no_source]==old['pred'][no_source])
en=np.abs(truth-new['pred']);eo=np.abs(truth-old['pred'])
from recuperar_subtrayectos import metric_xy
xy=metric_xy(d);chord=np.linalg.norm(xy[:,2:]-xy[:,:2],axis=1)
def summarize(mask,label):
 assert mask.any()
 return dict(label=label,n=int(mask.sum()),share=float(mask.mean()),
  model=metric(truth[mask],new['pred'][mask]),parent=metric(truth[mask],old['pred'][mask]),
  improvement_seconds_total=float((eo[mask]-en[mask]).sum()),fraction_changed=float(np.mean(new['pred'][mask]!=old['pred'][mask])),
  coverage80=float(np.mean((truth[mask]>=new['quantiles'][mask,0])&(truth[mask]<=new['quantiles'][mask,2]))),
  width80_s=float(np.mean(new['quantiles'][mask,2]-new['quantiles'][mask,0])))
source_groups=[summarize((counts>=lo)&(counts<hi),str(lo)+'..'+str(hi)) for lo,hi in [(0,1),(1,4),(4,8),(8,100)]]
duration_groups=[summarize((truth>=lo)&(truth<hi),str(lo)+'..'+str(hi)+'s') for lo,hi in [(60,300),(300,600),(600,900),(900,1800)]]
distance_groups=[summarize((chord>=lo)&(chord<hi),str(lo)+'..'+str(hi)+'m') for lo,hi in [(0,500),(500,1000),(1000,2000),(2000,20000)]]
for groups in [source_groups,duration_groups,distance_groups]:assert sum(v['n'] for v in groups)==127941
high=en>=300
result=dict(complete=True,primary_n=127941,no_original_queries_removed=True,
 structural_fallback=dict(no_sources_n=int(no_source.sum()),all_parent_points_exact=True),
 source_groups=source_groups,true_duration_groups=duration_groups,public_chord_groups=distance_groups,
 large_error=dict(error_threshold_s=300,n=int(high.sum()),share=float(high.mean()),fraction_of_all_absolute_error=float(en[high].sum()/en.sum())),
 point_changes=dict(n_changed=int(np.sum(new['pred']!=old['pred'])),n_better=int(np.sum(en<eo)),n_worse=int(np.sum(en>eo)),
  mean_change_s=float(np.mean(new['pred']-old['pred'])),mean_abs_change_s=float(np.mean(np.abs(new['pred']-old['pred'])))),
 causal_claim=False,selection_change=False,interpretation='Descriptive reusedValidation audit; trueTgroups are diagnostics, never queryfeatures or a narrowedcohort. No irreduciblebound.',
 no_validation_query_paths_decoded=True,Test_access=False,actual_data_files_accessed=sorted(access),
 sources_sha256=sha(__file__),seconds=time.time()-start)
write(OUT/'diagnostico_mejor_subtrayectos.json',result)
print(json.dumps(dict(primary_n=127941,changed=result['point_changes'],large_error=result['large_error'],
 fallback=result['structural_fallback'],duration_groups=[dict(label=v['label'],n=v['n'],MAE_s=v['model']['mae_s']) for v in duration_groups])),flush=True)
