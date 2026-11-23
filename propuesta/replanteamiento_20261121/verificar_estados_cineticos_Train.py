"""Bounded4096FITroadstate/clock/resourceinstrument. No newtimeMAE or populationbank."""
from base_replanteo import *
from estados_cineticos_15s import PublicRoadSampler,fitted_pair_distances
from coordenadas_trayecto_2d import metric_queries
start=time.time();access=deny_test()
run=OUT/'verificacion_estados_cineticos_Train_s0';assert not run.exists();run.mkdir()
with np.load(OUT/'auditoria_cinematica_ajuste_Train_s0/padres_ajuste_diagnostico.npz') as z:
 parents=z['parent'];rows=z['row'];timestamps=z['timestamp'];truth=z['source_time_s']
with np.load(OUT/'indices_escala_Train_s0/metadatos_Train.npz') as z:
 assert np.isin(parents,z['train_index'][:int(z['fit_n'])]).all()
with np.load(DATA/'train_x.npz') as d:
 assert np.array_equal(d['row'][parents],rows) and np.array_equal(d['timestamp'][parents],timestamps)
 queries={key:d[key][parents] for key in ['o_lon','o_lat','d_lon','d_lat','timestamp']}
with np.load(OUT/'memoria_puntos_Train_s0/puntos_Train.npz') as z:
 offsets=z['offsets'];n=z['parent_gps_n'][parents].astype(int)
 assert np.array_equal((n-1)*15,truth) and np.all(timestamps+truth<1396531073)
 original=z['xy_m'];points=np.concatenate([original[offsets[p]:offsets[p+1]] for p in parents]).astype(np.float64)
 del original
fit_offsets=np.r_[0,np.cumsum(n)];assert fit_offsets[-1]==len(points)
sampler=PublicRoadSampler()
print(json.dumps(dict(stage='road_sampler_ready',public_directed_edges=len(sampler.raw['u']),
 public_spatial50m_states=int(sampler.state_offsets[-1]),sampled_fitpoints=len(points),seconds=time.time()-start)),flush=True)
matched,nearest=sampler.match_fit(points,fit_offsets)
edge,fraction,error,resets=matched
assert len(edge)==len(points) and (edge>=0).all() and np.isfinite(error).all() and ((fraction>=0)&(fraction<=1)).all()
state=sampler.spatial_states(edge,fraction)
continues=np.ones(len(points)-1,bool);continues[fit_offsets[1:-1]-1]=False
displacement=np.sqrt((np.diff(points,axis=0)**2).sum(1))
road_distance=fitted_pair_distances(edge,fraction,continues,sampler.raw['u'],sampler.raw['v'],sampler.raw['length'],
 sampler.D,*sampler.connections)
# Sourcequality follows existingmapmatcher thresholds; no primaryquery or durationlabel is removed.
valid=continues&(displacement<=500)&(error[:-1]<=80)&(error[1:]<=80)&np.isfinite(road_distance)&(road_distance<=750)
moving=np.zeros(len(points),np.uint8)
for a,b in zip(fit_offsets[:-1],fit_offsets[1:]):
 d=displacement[a:b-1]
 velocity_class=np.where(d<=5,0,np.where(d<=150,1,2)).astype(np.uint8)
 moving[a]=velocity_class[0];moving[a+1:b]=velocity_class
kinetic=state*3+moving
unique_state=len(np.unique(kinetic))
pairs=np.column_stack([kinetic[:-1][valid],kinetic[1:][valid]])
unique_pairs=len(np.unique(pairs,axis=0))
probe={key:value[:65] for key,value in queries.items()}
public=sampler.public_attachments(probe)
reverse=sampler.public_attachments({key:value[::-1] for key,value in probe.items()})
single=sampler.public_attachments({key:value[:1] for key,value in probe.items()})
for a,b,c in zip(public,reverse,single):
 assert np.array_equal(a,b[::-1]) and np.array_equal(a[:1],c)
assert all(len(a)==65 and np.isfinite(a).all() for a in public)
# The fullsource HMM uses future points within a completedFITtrip. Its endpointedge labels are not queryinputs.
nearest_start=float(np.mean(edge[fit_offsets[:-1]]==nearest[fit_offsets[:-1]]))
nearest_end=float(np.mean(edge[fit_offsets[1:]-1]==nearest[fit_offsets[1:]-1]))
file=run/'estados_ajuste4096.npz'
np.savez_compressed(file,parent=parents,row=rows,timestamp=timestamps,original_time_s=truth,
 offsets=fit_offsets,edge=edge,fraction=fraction,error_m=error,kinetic_state=kinetic.astype(np.int64),
 valid_auxiliary_pair=valid,source_resets=resets)
assert sum(np.diff(fit_offsets)-1)==int(np.sum(truth/15))
proof=dict(passed=True,scope='4096precutoffFITstate/clock instrument; no predictiveMAE or newprimaryevaluation.',
 source_fitparents=4096,source_fitpoints=len(points),source15secondintervals=int(continues.sum()),
 clock_from_originalGPS_not_road_edgecount=True,all_original_T_labels_andqueries_preserved=True,
 all_sources_arrive_before1396531073=True,public_roadcell_maxwidth_m=50,
 motion_classes='Incoming15sdisplacement<=5m;5..150m;>150m. Initialclass inferredonlyfor completedFITsource, never supplied as actualqueryspeed.',
 public_directed_edges=len(sampler.raw['u']),public_spatial_states=int(sampler.state_offsets[-1]),
 theoretical_kinetic_states=int(3*sampler.state_offsets[-1]),sample_observed_kinetic_states=unique_state,
 sample_unique_valid_transitions=unique_pairs,
 valid_auxiliary_pair_fraction=float(valid.sum()/continues.sum()),
 source_projection_error_m_quantiles=np.quantile(error,[0,.5,.9,.99,1]),
 source_trips_with_HMMreset=int(np.sum(resets>0)),source_total_HMMresets=int(resets.sum()),
 nearest1_public_candidate_equals_source_HMM_start_fraction=nearest_start,
 nearest1_public_candidate_equals_source_HMM_end_fraction=nearest_end,
 public_raw5_65_reverse_singleton_all6arrays_exact=True,
 no_actualqueryroadID_speed_N_T_supplied=True,
 source_endpoint_inference_limit='HMMlabels depend on completedsourcepaths and are not annotatedgroundtruth. Queryattachments must integratepubliccandidates ratherthan use HMMfirst/last labels.',
 quality_limit='Pairs with mapmatcherresetlike geometry/jump/error are excludedonlyfrom auxiliarytransitioncounts; their trips and originaldurationtargets remain in everyprimary/heldevaluation.',
 large_populationbank_written=False,artifact_MiB=file.stat().st_size/1024**2,
 artifact_sha256=sha(file),next='Assess boundedprefixsourcecounts/sparseclock feasibility before any populationconstruction or timeexperiment.',
 Validation=False,Test=False,seconds=time.time()-start,
 actual_data_files_accessed=sorted(access),
 sources={name:sha(OUT/name) for name in ['verificar_estados_cineticos_Train.py','estados_cineticos_15s.py','coordenadas_trayecto_2d.py']},
 frozen_mapmatcher_sha256=sha(FROZEN/'map_matching.py'))
write(run/'resultado.json',proof)
for name in proof['sources']:(run/name).write_bytes((OUT/name).read_bytes())
print((run/'resultado.json').read_text(encoding='utf-8'),flush=True)
