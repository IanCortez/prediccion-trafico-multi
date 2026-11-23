"""Fixed4096fitparent kinetic audit; no modeltraining, heldtrajectory or MAE."""
from base_replanteo import *
start=time.time();access=deny_test()
run=OUT/'auditoria_cinematica_ajuste_Train_s0';assert not (run/'resultado.json').exists();run.mkdir(exist_ok=True)
indexmeta=read(OUT/'indices_escala_Train_s0/resultado.json')
with np.load(OUT/'indices_escala_Train_s0/indices_fuentes.npz') as z:
 fit_rows=z['row'][:indexmeta['fit_n']].copy()
bankfile=OUT/'memoria_puntos_Train_s0/puntos_Train.npz'
assert sha(bankfile)==read(OUT/'memoria_puntos_Train_s0/resultado.json')['outputs_sha256']
with np.load(bankfile) as z:
 row=z['row'];ts=z['timestamp'];n=z['parent_gps_n'];offsets=z['offsets']
 eligible=np.isin(np.arange(len(n)),z['eligible'])&(n>0)&np.isin(row,fit_rows)&(ts+15*(n.astype(np.int64)-1)<indexmeta['source_cutoff'])
 candidates=np.flatnonzero(eligible)
 chosen=np.sort(np.random.default_rng(20261129).choice(candidates,4096,replace=False))
 assert np.isin(row[chosen],fit_rows).all()
 # Only selected fitparent coordinates enter diagnostics; no heldpoint summaries or graph.
 xy=z['xy_m']
 speed=[];angles=[];trip=[];cells=[];edges=[];jumps=[];previous=0
 for p in chosen:
  points=xy[offsets[p]:offsets[p+1]].astype(np.float64)
  delta=np.diff(points,axis=0);length=np.sqrt((delta**2).sum(1));speed.extend((length/15*3.6).tolist())
  if len(length)>1:
   good=(length[:-1]>15)&(length[1:]>15)
   cosine=(delta[:-1]*delta[1:]).sum(1)/np.maximum(length[:-1]*length[1:],1e-15)
   angles.extend(np.clip(cosine[good],-1,1).tolist())
  # Fixed100m grid is a feasibility diagnostic, never an evaluationfilter or a tunedhyperparameter.
  grid=np.floor(points/100).astype(np.int32);cells.append(grid)
  paired_cells=np.column_stack([grid[:-1],grid[1:]]);edges.append(paired_cells)
  jumps.extend(np.max(np.abs(np.diff(grid,axis=0)),axis=1).tolist())
  trip.append([row[p],ts[p],15*(len(points)-1),float(np.mean(length<=5)),float(np.mean(length==0)),
   float(np.mean((grid[:-1]==grid[1:]).all(1))),float(np.max(length))])
 del xy
trip=np.asarray(trip,dtype=np.float64);speed=np.asarray(speed);angles=np.asarray(angles);jumps=np.asarray(jumps)
grid=np.concatenate(cells);transition=np.concatenate(edges)
unique_cells=np.unique(grid,axis=0);unique_edges=np.unique(transition,axis=0)
quant=lambda a:dict(zip(['p0','p50','p90','p99','max'],np.quantile(a,[0,.5,.9,.99,1]).tolist()))
groups=[]
for lo,hi in [(60,300),(300,600),(600,900),(900,1800)]:
 a=trip[(trip[:,2]>=lo)&(trip[:,2]<hi)]
 groups.append(dict(trueT_range_s=[lo,hi],n=len(a),mean_fraction_displacement_le5m=float(a[:,3].mean()),mean_same100mcell_fraction=float(a[:,5].mean())))
np.savez_compressed(run/'padres_ajuste_diagnostico.npz',parent=chosen,row=trip[:,0].astype(np.int64),
 timestamp=trip[:,1].astype(np.int64),source_time_s=trip[:,2],fraction_displacement_le5m=trip[:,3],fraction_exact_stationary=trip[:,4],
 fraction_same100mcell=trip[:,5],max_step_displacement_m=trip[:,6])
result=dict(complete=True,scope='4096 deterministicprecutofffitparents only; no predictiveexperiment or newMAE.',
 seed=20261129,eligible_fitparents=len(candidates),sampled_fitparents=4096,sampled15secondtransitions=len(speed),
 selected_parent_arrivals_before=indexmeta['source_cutoff'],all_sampled_rows_in_original951718fit=True,
 physical_speed_kmh_quantiles=quant(speed),fraction_steps_gt120kmh=float(np.mean(speed>120)),
 adjacent_moving_directions_cosine_quantiles=quant(angles),fraction_moving_direction_change_gt90deg=float(np.mean(angles<0)),
 fixed100m_grid=dict(visited_cells=len(unique_cells),observed_directed_jumps=len(unique_edges),
  largest_celljump_quantiles=quant(jumps),samecell_fraction=float(np.mean(jumps==0)),
  fraction_jump_outside_3by3_neighborhood=float(np.mean(jumps>1))),
 trip_groups=groups,
 interpretation_limits=[
 'Displacement<=5m is a GPS displacementproxy, not verified vehicleidling or passengerbehavior.',
 'Same100mcell mixes truewaiting with movement inside a cell; it is not a physicalstop label.',
 'Observedspatialresolution/directionpersistence do not establish a timepredictionimprovement.',
 'The public static53766roadchain adjacency has variablelength edges; one roadedge must NEVER be treated as one15second tick.',
 'A graph of15secondGPStransitions would need fit-only construction and publicraw5 endpointprojection. This audit stores no extraGPSbank or graph.',
 'No sourcekinematic anomaly removes a primaryquery or changes its original target.'
 ],predictions_written=False,original_cohort_and_targets_unchanged=True,
 Validation=False,Test=False,actual_data_files_accessed=sorted(access),source_sha256=sha(__file__),
 provenance_sha256=sha(run/'padres_ajuste_diagnostico.npz'),seconds=time.time()-start)
write(run/'resultado.json',result);(run/'fuente_ejecutada.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(result))
