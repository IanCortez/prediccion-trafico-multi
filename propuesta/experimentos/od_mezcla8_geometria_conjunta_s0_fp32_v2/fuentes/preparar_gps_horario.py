from common import *
from features import *
from ruteo_compacto import *
from perfiles_ruta import *
import torch,sys
from numba import set_num_threads
sys.path.insert(0,str(ROOT/'codigo-ian/src'));import tte_model
started=time.time();set_num_threads(3)
raw=dict(np.load(DATA/'mapa_dirigido.npz'));comp=dict(np.load(DATA/'grafo_compacto.npz'));q=dict(np.load(DATA/'consultas_ruteo_compacto.npz'))
cd=torch.load(DATA/'original_completo_gps_state_v1.pt',map_location='cpu',weights_only=False)
spd=cd.st.spd.numpy();obs=cd.st.obs.numpy();fold=cd.st.fold.numpy().astype(np.int64);assert spd.shape==(3,168,253)
allx=[dict(np.load(DATA/(s+'_x.npz'))) for s in ('train','validation','test')];F=np.concatenate([features(d)[0] for d in allx]);how=F[:,15].astype(np.int64)*24+F[:,14].astype(np.int64)
fold=fold[:len(allx[0]['row'])];assert len(fold)==len(allx[0]['row']) and np.isin(fold,[0,1]).all()
profile=np.r_[fold,np.full(len(how)-len(fold),2,np.int64)]*168+how
assert profile.min()>=0 and profile.max()<504
mid=(raw['xy'][raw['u']]+raw['xy'][raw['v']])/2;lon=ORIGIN[0]+mid[:,0]/(111195*np.cos(np.deg2rad(ORIGIN[1])));lat=ORIGIN[1]+mid[:,1]/111195
inside=(lat>=41.1405)&(lat<=41.1865)&(lon>=-8.6887)&(lon<=-8.5557)
dlat=500/110540;dlon=500/(111320*np.cos(np.deg2rad((41.1405+41.1865)/2)));cell=(np.floor((lat-41.1405)/dlat).astype(int)*23+np.floor((lon+8.6887)/dlon).astype(int)).clip(0,252)
speed=np.clip((15/3.6)*np.exp(spd.reshape(504,253)[:,cell]*np.log(2)),2/3.6,120/3.6)
time_arc=np.where(inside[None,:],raw['length'][None,:]/speed,raw['length'][None,:]/(20/3.6)).astype(np.float32)
support_arc=(raw['length'][None,:]*obs.reshape(504,253)[:,cell]*inside[None,:]).astype(np.float32)
pt,pc,ct,cc=chain_profiles(comp['chain_offsets'],comp['flat_incoming_arc'],np.ascontiguousarray(time_arc.T),np.ascontiguousarray(support_arc.T));del speed,cd
outputs=[];names=[];checks={};N=len(q['row']);snap=q['o_snap']+q['d_snap']
for kind in ('length','free'):
    graph,chosen=compressed_csr(comp,kind);pred=np.load(DATA/f'compact_{kind}_pred.npy',mmap_mode='r');choice=np.load(DATA/f'compact_{kind}_choices.npz')['choice']
    ids=np.flatnonzero((choice>=0)&(choice<4));oc=np.maximum(choice,0)//2;dc=np.maximum(choice,0)%2;sr=np.full(N,-1,np.int32);dest=np.full(N,-1,np.int32)
    sr[ids]=q['source_lookup'][q['o_junction'][ids,oc[ids]]];dest[ids]=q['d_junction'][ids,dc[ids]];order=ids[np.argsort(sr[ids],kind='stable')];bounds=np.searchsorted(sr[order],np.arange(len(q['sources'])+1))
    result=routed_profiles(pred,q['sources'],graph.indptr,graph.indices,chosen,ct,cc,comp['chain_length'],comp['chain_free'],order,bounds,dest,profile,N)
    for o in range(2):
        for d in range(2):
            ix=np.flatnonzero(choice==o*2+d)
            for tag,slot in (('o',o),('d',d)):
                cid=q[tag+'_chain'][ix,slot];pos=q[tag+'_pos'][ix,slot];valid=cid>=0;use=ix[valid];c=cid[valid];p=pos[valid];pr=profile[use]
                result[use,0]+=(ct[c,pr]-pt[p,pr]) if tag=='o' else pt[p,pr];result[use,1]+=(cc[c,pr]-pc[p,pr]) if tag=='o' else pc[p,pr]
            result[ix,2]+=q['o_length'][ix,o]+q['d_length'][ix,d];result[ix,3]+=q['o_free'][ix,o]+q['d_free'][ix,d]
            ix=np.flatnonzero(choice==4+o*2+d);op=q['o_pos'][ix,o];dp=q['d_pos'][ix,d];pr=profile[ix]
            result[ix,0]=pt[dp,pr]-pt[op,pr];result[ix,1]=pc[dp,pr]-pc[op,pr];result[ix,2]=comp['prefix_length'][dp]-comp['prefix_length'][op];result[ix,3]=comp['prefix_free'][dp]-comp['prefix_free'][op]
    good=(choice>=0)&(q['o_snap']<500)&(q['d_snap']<500)&np.isfinite(result).all(1);result[:,0]+=snap/(20/3.6);result[:,2]+=snap;result[:,3]+=snap/(20/3.6)
    result[~good,0]=60+F[~good,6]*1000/(20/3.6);result[~good,1]=0;result[~good,2]=F[~good,6]*1000;result[~good,3]=result[~good,2]/(25/3.6)
    X=np.column_stack([result[:,0]/600,result[:,2]/1000,result[:,3]/600,np.clip(result[:,1]/np.maximum(result[:,2],1),0,1),np.log(np.maximum(result[:,0]/np.maximum(result[:,3],1),.01)),np.log(np.maximum(1000*result[:,0]/np.maximum(result[:,2],1),.01)),good.astype(float)]).astype(np.float32)
    assert np.isfinite(X).all();outputs.append(X);names.extend(['hourly_'+kind+'_'+s for s in ('gps_time_600s','path_km','freeflow_600s','observed_fraction','congestion_log','pace_log','available')]);checks[kind]=dict(available=int(good.sum()))
    print(dict(stage=kind,seconds=time.time()-started),flush=True)
X=np.column_stack(outputs)
for i,split in enumerate(('train','validation','test')):
    lo,hi=q['split_bounds'][i:i+2];path=DATA/(split+'_gps_horario.npz');assert not path.exists(),path;np.savez_compressed(path,row=q['row'][lo:hi],X=X[lo:hi],names=np.array(names))
np.savez_compressed(DATA/'arcos_gps_horario.npz',time_s=time_arc,observed_length_m=support_arc,query_profile=profile)
write_json(DATA/'gps_horario_manifiesto.json',dict(features=names,source_state_sha256=sha(DATA/'original_completo_gps_state_v1.pt'),arc_profiles_sha256=sha(DATA/'arcos_gps_horario.npz'),state_units='log actual GPS speed m/s divided by 15 km/h, base 2',profiles=504,clock='Exact Europe/Lisbon departure hour of week',grid_m=500,folds='Two original fixed folds, query fold excluded; full training GPS for validation/test',pseudo_distance_m=500,pseudo_seconds=120,observed_fraction='Binary observation by cell/hour; not a statistical confidence estimate',no_heading=True,test_labels_read=False,seconds=time.time()-started))
print(dict(stage='done',seconds=time.time()-started),flush=True)

