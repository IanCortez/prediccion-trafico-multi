from common import *
from features import *
from ruteo_compacto import *
from geometria_ruteada import sample_routes
from numba import set_num_threads
started=time.time();set_num_threads(4)
raw=dict(np.load(DATA/'mapa_dirigido.npz'));comp=dict(np.load(DATA/'grafo_compacto.npz'));q=dict(np.load(DATA/'consultas_ruteo_compacto.npz'))
records=[dict(np.load(DATA/(s+'_x.npz'))) for s in ('train','validation','test')]
F=np.concatenate([features(d)[0] for d in records]);orig=F[:,:2].astype(np.float64)*1000;dest=F[:,2:4].astype(np.float64)*1000
outputs=[];names=[];checks={}
for kind in ('length','free'):
    graph,chosen=compressed_csr(comp,kind);choice=np.load(DATA/f'compact_{kind}_choices.npz')['choice']
    pred=np.load(DATA/f'compact_{kind}_pred.npy',mmap_mode='r')
    X=sample_routes(raw['xy'],comp['flat_nodes'],comp['chain_offsets'],comp['prefix_length'],comp['junction_raw'],graph.indptr,graph.indices,chosen,pred,q['source_lookup'],choice,q['o_chain'],q['d_chain'],q['o_pos'],q['d_pos'],q['o_junction'],q['d_junction'],orig,dest)
    good=(choice>=0)&(q['o_snap']<500)&(q['d_snap']<500)
    # A straight-line physical fallback has zero deviations and zero turns.
    X[~good]=0;X[~good,14]=F[~good,6]
    previous=np.concatenate([np.load(DATA/(s+'_gps_rutas.npz'))['X'][:,1 if kind=='length' else 8] for s in ('train','validation','test')])
    err=np.abs(X[good,14]-previous[good]);assert err.max()<.002,(kind,float(err.max()))
    assert np.isfinite(X).all()
    names.extend([kind+'_shape_'+str(i) for i in range(14)]+[kind+'_'+v for v in ('polyline_km','polyline_circuity_log','turn_abs_pi','turn_signed_pi','sharp_turn_count','max_turn_pi','shape_available')]);outputs.append(X)
    checks[kind]=dict(max_length_disagreement_m=float(err.max()*1000),available=int(good.sum()))
    print(dict(kind=kind,checks=checks[kind],seconds=time.time()-started),flush=True)
X=np.column_stack(outputs)
for i,split in enumerate(('train','validation','test')):
    lo,hi=q['split_bounds'][i:i+2];path=DATA/(split+'_geometria_rutas.npz');assert not path.exists(),path
    np.savez_compressed(path,row=q['row'][lo:hi],X=X[lo:hi],names=np.array(names))
write_json(DATA/'geometria_rutas_manifiesto.json',dict(features=names,checks=checks,graph_sha256=sha(DATA/'grafo_compacto.npz'),query_sha256=sha(DATA/'consultas_ruteo_compacto.npz'),seconds=time.time()-started,test_labels_read=False,inputs='Endpoints and static directed map only; seven normalized-arclength interior points per candidate; never observed evaluated trajectory.'))
