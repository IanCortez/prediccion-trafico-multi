from common import *
from map_matching import *
from scipy.spatial import cKDTree
from numba import set_num_threads
started=time.time();set_num_threads(4)
contract=json.loads((BASE/'instrumentos/map_matching_contrato.json').read_text());assert contract['passed']
raw=dict(np.load(DATA/'mapa_dirigido_2014.npz'));comp=dict(np.load(DATA/'grafo_compacto_2014.npz'));distance_info=json.loads((DATA/'distancias_2014/manifiesto.json').read_text());assert distance_info['raw_graph_sha256']==sha(DATA/'mapa_dirigido_2014.npz');D=np.load(DATA/'distancias_2014/D.npy',mmap_mode='r');connections=node_connections(comp)
z=dict(np.load(DATA/'trayectorias_train.npz'));ntrips=len(z['row']);npoints=len(z['xy']);batch_trips=10000;chunks=int(np.ceil(ntrips/batch_trips));folder=DATA/'map_matching_2014_v2';folder.mkdir(exist_ok=True)
config=dict(implementation_version=2,source_sha256=sha(DATA/'trayectorias_train.npz'),map_sha256=distance_info['raw_graph_sha256'],distance_sha256=distance_info['distance_sha256'],point_sampling_m=25,nearest_samples=32,candidates=4,gps_sigma_m=30,transition_beta='50 + 0.1 * GPS step distance in m',maximum_transition_m=750,reset_GPS_jump_m=500,batch_trips=batch_trips,n_trips=ntrips,n_points=npoints)
if (folder/'config.json').exists():
    assert json.loads((folder/'config.json').read_text())==config;matched=np.load(folder/'edge.npy',mmap_mode='r+');fraction=np.load(folder/'fraction.npy',mmap_mode='r+');error=np.load(folder/'error_m.npy',mmap_mode='r+');reset=np.load(folder/'resets.npy',mmap_mode='r+');done=np.load(folder/'done.npy',mmap_mode='r+')
else:
    write_json(folder/'config.json',config);matched=np.lib.format.open_memmap(folder/'edge.npy',mode='w+',dtype=np.int32,shape=(npoints,));fraction=np.lib.format.open_memmap(folder/'fraction.npy',mode='w+',dtype=np.float32,shape=(npoints,));error=np.lib.format.open_memmap(folder/'error_m.npy',mode='w+',dtype=np.float32,shape=(npoints,));reset=np.lib.format.open_memmap(folder/'resets.npy',mode='w+',dtype=np.int32,shape=(ntrips,));done=np.lib.format.open_memmap(folder/'done.npy',mode='w+',dtype=bool,shape=(chunks,));done[:]=False
count=np.maximum(1,np.ceil(raw['length']/25).astype(np.int32));sample_edge=np.repeat(np.arange(len(count),dtype=np.int32),count);local=np.arange(len(sample_edge))-np.repeat(np.cumsum(count)-count,count);t=(local+.5)/count[sample_edge];a=raw['xy'][raw['u'][sample_edge]];b=raw['xy'][raw['v'][sample_edge]];sampling=a+(b-a)*t[:,None];tree=cKDTree(sampling)
print(dict(stage='loaded',trips=ntrips,points=npoints,road_samples=len(sampling),seconds=time.time()-started),flush=True)
for chunk,first in enumerate(range(0,ntrips,batch_trips)):
    if done[chunk]:continue
    last=min(first+batch_trips,ntrips);lo,hi=z['offsets'][[first,last]];points=z['xy'][lo:hi].astype(float)*1000
    phase=time.time();distances,near=tree.query(points,k=32,workers=4);del distances;kd_seconds=time.time()-phase;phase=time.time()
    edges,frac,err=project_candidates(points,near,sample_edge,raw['xy'],raw['u'],raw['v']);del near;assert np.all(edges>=0);project_seconds=time.time()-phase;phase=time.time()
    result=viterbi_match(points,z['offsets'][first:last+1]-lo,edges,frac,err,raw['u'],raw['v'],raw['length'],D,*connections)
    viterbi_seconds=time.time()-phase
    matched[lo:hi]=result[0];fraction[lo:hi]=result[1];error[lo:hi]=result[2];reset[first:last]=result[3]
    matched.flush();fraction.flush();error.flush();reset.flush();done[chunk]=True;done.flush()
    if True:print(dict(kd_seconds=kd_seconds,project_seconds=project_seconds,viterbi_seconds=viterbi_seconds,trips=f'{last}/{ntrips}',points=int(hi),median_projection_error_m=float(np.median(result[2])),resets=int(result[3].sum()),seconds=time.time()-started),flush=True)
assert done.all() and np.all(matched>=0) and np.all((fraction>=0)&(fraction<=1)) and np.isfinite(error).all()
sample=np.asarray(error[::97]);summary=dict(config=config,projection_error_quantiles_m=dict(zip(['p25','p50','p90','p95','p99'],np.quantile(sample,[.25,.5,.9,.95,.99]))),sampled_fraction_within30m=float(np.mean(sample<=30)),sampled_fraction_within80m=float(np.mean(sample<=80)),trips_with_reset=int(np.sum(reset>0)),resets=int(reset.sum()),test_labels_read=False,training_only=True,caveat='Projection residual and reset counts measure algorithmic fit, not annotated map-matching accuracy. These GPS trajectories are training resources, never evaluated-query inputs.',code_hashes={name:sha(BASE/name) for name in ('map_matching.py','preparar_map_matching.py','compresion_vial.py')},output_hashes={name:sha(folder/name) for name in ('edge.npy','fraction.npy','error_m.npy','resets.npy')},seconds=time.time()-started)
write_json(folder/'manifiesto.json',summary);print(dict(stage='done',summary=summary),flush=True)

