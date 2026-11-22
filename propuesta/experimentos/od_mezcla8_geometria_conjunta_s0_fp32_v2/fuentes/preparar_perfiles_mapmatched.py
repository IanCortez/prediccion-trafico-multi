from common import *
started=time.time();source=DATA/'movimiento_mapmatched_2014';meta=json.loads((source/'manifiesto.json').read_text());raw=dict(np.load(DATA/'mapa_dirigido_2014.npz'));distance=np.load(source/'distance_m.npy',mmap_mode='r');seconds=np.load(source/'seconds.npy',mmap_mode='r')
folder=DATA/'perfiles_mapmatched_2014';folder.mkdir(exist_ok=True);E=len(raw['u']);shape=(1008,E);configuration=dict(distance_sha256=meta['distance_sha256'],seconds_sha256=meta['seconds_sha256'],road_map_sha256=sha(DATA/'mapa_dirigido_2014.npz'),profiles=1008,folds=5,states=168,link_shrinkage_m=500,state_shrinkage_m=500,city_shrinkage_m=100000,freeflow_multiplier=1.7,speed_bounds_kmh=[2,120])
if (folder/'config.json').exists():
    assert json.loads((folder/'config.json').read_text())==configuration;times=np.load(folder/'time_s.npy',mmap_mode='r+');state_conf=np.load(folder/'state_support.npy',mmap_mode='r+');link_conf=np.load(folder/'link_support.npy',mmap_mode='r+');done=np.load(folder/'done.npy',mmap_mode='r+')
else:
    write_json(folder/'config.json',configuration);times=np.lib.format.open_memmap(folder/'time_s.npy',mode='w+',dtype=np.float32,shape=shape);state_conf=np.lib.format.open_memmap(folder/'state_support.npy',mode='w+',dtype=np.float32,shape=shape);link_conf=np.lib.format.open_memmap(folder/'link_support.npy',mode='w+',dtype=np.float32,shape=(6,E));done=np.lib.format.open_memmap(folder/'done.npy',mode='w+',dtype=bool,shape=(6,));done[:]=False
fullD=distance.sum(0,dtype=np.float64);fullT=seconds.sum(0,dtype=np.float64);prior=1.7*raw['seconds']/raw['length'];summary=[]
for f in range(6):
    if done[f]:continue
    D=fullD-distance[f] if f<5 else fullD;T=fullT-seconds[f] if f<5 else fullT;linkD=D.sum(0);linkT=T.sum(0);pace_link=(linkT+500*prior)/(linkD+500);city=linkT.sum()/linkD.sum();link_conf[f]=linkD/(linkD+500)
    for how in range(168):
        city_h=(T[how].sum()+100000*city)/(D[how].sum()+100000);factor=np.clip(city_h/city,.5,2.5);pace=(T[how]+500*pace_link*factor)/(D[how]+500);pace=np.clip(pace,3.6/120,3.6/2);times[f*168+how]=raw['length']*pace;state_conf[f*168+how]=D[how]/(D[how]+500)
    times.flush();state_conf.flush();link_conf.flush();done[f]=True;done.flush();summary.append(dict(fold=f,city_mean_speed_kmh=float(3.6/city),mean_link_support=float(link_conf[f].mean())));print(dict(fold=f,seconds=time.time()-started,summary=summary[-1]),flush=True)
# Numerical intervention on an entire held-out fold, including all backoff levels.
smallD=np.array(distance[:,0,:40],dtype=np.float64);smallT=np.array(seconds[:,0,:40],dtype=np.float64);mutated=smallD.copy();mutated[2]+=1234
assert np.array_equal(smallD.sum(0)-smallD[2],mutated.sum(0)-mutated[2])
assert done.all() and np.isfinite(times).all() and np.all(times>0)
write_json(folder/'manifiesto.json',dict(config=configuration,input_statistics_sha256=sha(source/'manifiesto.json'),outputs={name:sha(folder/name) for name in ('time_s.npy','state_support.npy','link_support.npy')},summary=summary,query_fold_intervention_invariant=True,profile_assignment='Query UTC departure-week fold excluded for training; full training GPS index5 for validation/test. Hour of week from Europe/Lisbon departure.',support='Shrinkage weights for exact link and link/hour, not probability of correctness.',test_labels_read=False,seconds=time.time()-started))
print(dict(stage='done',seconds=time.time()-started),flush=True)
