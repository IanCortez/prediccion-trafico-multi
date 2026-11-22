from common import *
from features import *
from ruteo_compacto import *
from geometria_ruteada import sample_routes
from perfiles_ruta import routed_profiles
from scipy.sparse.csgraph import dijkstra
from numba import set_num_threads
started=time.time();set_num_threads(3)
raw=dict(np.load(DATA/'mapa_dirigido.npz'));comp=dict(np.load(DATA/'grafo_compacto.npz'));q=dict(np.load(DATA/'consultas_ruteo_compacto.npz'))
p=dict(np.load(DATA/'prefijos_movimiento_compacto.npz'));pt=p['prefix_time'];pc=p['prefix_support'];ct=p['chain_time'];cc=p['chain_support']
records=[dict(np.load(DATA/(s+'_x.npz'))) for s in ('train','validation','test')];F=np.concatenate([features(d)[0] for d in records])
orig=F[:,:2].astype(float)*1000;dest=F[:,2:4].astype(float)*1000
checkpoint=DATA/'rutas_dinamicas_perfiles';checkpoint.mkdir(exist_ok=True)
N=len(q['row']);n=len(comp['junction_raw']);pair=comp['chain_src'].astype(np.int64)*n+comp['chain_dst'];checks=[]
names=['dynamic_shape_'+str(i) for i in range(14)]+['dynamic_'+v for v in ('polyline_km','polyline_circuity_log','turn_abs_pi','turn_signed_pi','sharp_turn_count','max_turn_pi','shape_available','gps_time_600s','support_fraction','freeflow_600s','congestion_log','pace_log')]
for profile in range(ct.shape[1]):
    target=checkpoint/f'profile_{profile:02d}.npz'
    if target.exists():continue
    ids=np.flatnonzero(q['profile_id']==profile);weight=ct[:,profile]
    order=np.lexsort((weight,pair));ordered=pair[order];chosen=order[np.r_[True,np.diff(ordered)!=0]]
    graph=csr_matrix((weight[chosen].astype(float),(comp['chain_src'][chosen],comp['chain_dst'][chosen])),shape=(n,n))
    sub={k:q[k][ids] for k in ('o_junction','d_junction','o_chain','d_chain','o_pos','d_pos','o_length','d_length','o_free','d_free','o_snap','d_snap')}
    sources=np.unique(sub['o_junction'][sub['o_junction']>=0]);lookup=np.full(n,-1,np.int32);lookup[sources]=np.arange(len(sources))
    dist,pred=dijkstra(graph,directed=True,indices=sources,return_predecessors=True)
    best=np.full(len(ids),np.inf);choice=np.full(len(ids),-1,np.int8)
    def endpoint(tag,slot):
        chain=sub[tag+'_chain'][:,slot];pos=sub[tag+'_pos'][:,slot];valid=chain>=0;cost=np.zeros(len(ids))
        cost[valid]=ct[chain[valid],profile]-pt[pos[valid],profile] if tag=='o' else pt[pos[valid],profile]
        cost[sub[tag+'_junction'][:,slot]<0]=np.inf
        return cost
    for os in range(2):
        for ds in range(2):
            source=sub['o_junction'][:,os];destination=sub['d_junction'][:,ds];valid=(source>=0)&(destination>=0);ix=np.flatnonzero(valid)
            cand=dist[lookup[source[ix]],destination[ix]]+endpoint('o',os)[ix]+endpoint('d',ds)[ix]
            good=cand<best[ix];use=ix[good];best[use]=cand[good];choice[use]=os*2+ds
            same=(sub['o_chain'][:,os]>=0)&(sub['o_chain'][:,os]==sub['d_chain'][:,ds])&(sub['d_pos'][:,ds]>=sub['o_pos'][:,os]);ix=np.flatnonzero(same)
            cand=pt[sub['d_pos'][ix,ds],profile]-pt[sub['o_pos'][ix,os],profile];good=cand<best[ix];use=ix[good];best[use]=cand[good];choice[use]=4+os*2+ds
    shape=sample_routes(raw['xy'],comp['flat_nodes'],comp['chain_offsets'],comp['prefix_length'],comp['junction_raw'],graph.indptr,graph.indices,chosen,pred,lookup,choice,sub['o_chain'],sub['d_chain'],sub['o_pos'],sub['d_pos'],sub['o_junction'],sub['d_junction'],orig[ids],dest[ids])
    normal=(choice>=0)&(choice<4);ix=np.flatnonzero(normal);os=np.maximum(choice,0)//2;ds=np.maximum(choice,0)%2;sr=np.full(len(ids),-1,np.int32);destination=np.full(len(ids),-1,np.int32)
    sr[ix]=lookup[sub['o_junction'][ix,os[ix]]];destination[ix]=sub['d_junction'][ix,ds[ix]]
    ordered=ix[np.argsort(sr[ix],kind='stable')];bounds=np.searchsorted(sr[ordered],np.arange(len(sources)+1))
    total=routed_profiles(pred,sources,graph.indptr,graph.indices,chosen,ct[:,profile:profile+1],cc[:,profile:profile+1],comp['chain_length'],comp['chain_free'],ordered,bounds,destination,np.zeros(len(ids),np.int64),len(ids))
    for os in range(2):
        for ds in range(2):
            ix=np.flatnonzero(choice==os*2+ds)
            for tag,slot in (('o',os),('d',ds)):
                chain=sub[tag+'_chain'][ix,slot];pos=sub[tag+'_pos'][ix,slot];valid=chain>=0;use=ix[valid];c=chain[valid];pos=pos[valid]
                total[use,0]+=(ct[c,profile]-pt[pos,profile]) if tag=='o' else pt[pos,profile]
                total[use,1]+=(cc[c,profile]-pc[pos,profile]) if tag=='o' else pc[pos,profile]
            total[ix,2]+=sub['o_length'][ix,os]+sub['d_length'][ix,ds];total[ix,3]+=sub['o_free'][ix,os]+sub['d_free'][ix,ds]
            ix=np.flatnonzero(choice==4+os*2+ds);op=sub['o_pos'][ix,os];dp=sub['d_pos'][ix,ds]
            total[ix,0]=pt[dp,profile]-pt[op,profile];total[ix,1]=pc[dp,profile]-pc[op,profile];total[ix,2]=comp['prefix_length'][dp]-comp['prefix_length'][op];total[ix,3]=comp['prefix_free'][dp]-comp['prefix_free'][op]
    good=(choice>=0)&(sub['o_snap']<500)&(sub['d_snap']<500)&np.isfinite(total).all(1)
    err=float(np.max(np.abs(total[good,0]-best[good]))) if good.any() else 0.;assert err<.1,(profile,err)
    snap=sub['o_snap']+sub['d_snap'];total[:,0]+=snap/(20/3.6);total[:,2]+=snap;total[:,3]+=snap/(20/3.6)
    total[~good,0]=60+F[ids[~good],6]*1000/(20/3.6);total[~good,1]=0;total[~good,2]=F[ids[~good],6]*1000;total[~good,3]=total[~good,2]/(25/3.6)
    shape[~good]=0;shape[~good,14]=F[ids[~good],6]
    assert np.max(np.abs(shape[good,14]*1000-total[good,2]))<.02
    X=np.column_stack([shape,total[:,0]/600,np.clip(total[:,1]/np.maximum(total[:,2],1),0,1),total[:,3]/600,np.log(np.maximum(total[:,0]/np.maximum(total[:,3],1),.01)),np.log(np.maximum(1000*total[:,0]/np.maximum(total[:,2],1),.01))]).astype(np.float32)
    assert np.isfinite(X).all() and X.shape[1]==len(names)
    np.savez_compressed(target,query_id=ids,X=X,max_cost_error_s=err)
    print(dict(profile=profile,n=len(ids),sources=len(sources),max_cost_error_s=err,seconds=time.time()-started),flush=True)
    del dist,pred,graph,total,shape,X
X=np.empty((N,len(names)),np.float32);seen=np.zeros(N,np.uint8)
for profile in range(ct.shape[1]):
    z=np.load(checkpoint/f'profile_{profile:02d}.npz');ids=z['query_id'];X[ids]=z['X'];seen[ids]+=1;checks.append(dict(profile=profile,max_cost_error_s=float(z['max_cost_error_s']),n=len(ids)))
assert np.all(seen==1)
for i,split in enumerate(('train','validation','test')):
    lo,hi=q['split_bounds'][i:i+2];path=DATA/(split+'_rutas_dinamicas.npz');assert not path.exists(),path
    np.savez_compressed(path,row=q['row'][lo:hi],X=X[lo:hi],names=np.array(names))
write_json(DATA/'rutas_dinamicas_manifiesto.json',dict(features=names,checks=checks,profile_sha256=sha(DATA/'arcos_tiempos_gps.npz'),query_sha256=sha(DATA/'consultas_ruteo_compacto.npz'),seconds=time.time()-started,test_labels_read=False,inference='Minimum empirical historical GPS-cost route for the departure state. Five UTC-week folds excluding query fold in train, full training history in validation/test; no observed query route.'))
