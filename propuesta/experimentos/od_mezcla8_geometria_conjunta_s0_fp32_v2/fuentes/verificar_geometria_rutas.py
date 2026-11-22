from common import *
from features import *
from ruteo_compacto import *
raw=dict(np.load(DATA/'mapa_dirigido.npz'));comp=dict(np.load(DATA/'grafo_compacto.npz'));q=dict(np.load(DATA/'consultas_ruteo_compacto.npz'))
F=np.concatenate([features(dict(np.load(DATA/(s+'_x.npz'))))[0] for s in ('train','validation','test')]);checks={}
for kind in ('length','free'):
    graph,chosen=compressed_csr(comp,kind);pred=np.load(DATA/f'compact_{kind}_pred.npy',mmap_mode='r');choice=np.load(DATA/f'compact_{kind}_choices.npz')['choice']
    out=np.concatenate([np.load(DATA/(s+'_geometria_rutas.npz'))['X'][:,0:21 if kind=='length' else 42] for s in ('train','validation','test')])
    if kind=='free':out=out[:,21:]
    errors=[]
    rng=np.random.default_rng(715)
    selected=rng.choice(np.flatnonzero(out[:,20]>0),100,replace=False)
    for code in range(8):
        ids=np.flatnonzero((choice==code)&(out[:,20]>0))
        if len(ids):selected=np.r_[selected,ids[:3]]
    for i in selected:
        c=int(choice[i]);os=(c%4)//2;ds=c%2;nodes=[]
        def chain_nodes(chain,lo=None,hi=None):
            if lo is None:lo=comp['chain_offsets'][chain]
            if hi is None:hi=comp['chain_offsets'][chain+1]-1
            return comp['flat_nodes'][lo:hi+1].tolist()
        if c>=4:nodes=chain_nodes(q['o_chain'][i,os],q['o_pos'][i,os],q['d_pos'][i,ds])
        else:
            chain=q['o_chain'][i,os]
            if chain>=0:nodes.extend(chain_nodes(chain,q['o_pos'][i,os]))
            root=q['o_junction'][i,os];node=q['d_junction'][i,ds];back=[];sr=q['source_lookup'][root]
            while node!=root:
                parent=pred[sr,node];edge=graph.indptr[parent]+np.flatnonzero(graph.indices[graph.indptr[parent]:graph.indptr[parent+1]]==node)[0]
                back.append(chosen[edge]);node=parent
            for chain in reversed(back):nodes.extend(chain_nodes(chain))
            chain=q['d_chain'][i,ds]
            if chain>=0:nodes.extend(chain_nodes(chain,hi=q['d_pos'][i,ds]))
            if not nodes:nodes=[comp['junction_raw'][root]]
        points=np.vstack([F[i,:2].astype(float)*1000,raw['xy'][nodes],F[i,2:4].astype(float)*1000])
        keep=np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-6];points=points[keep]
        lengths=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        fraction=np.arange(1,8)/8;pts=np.column_stack([np.interp(fraction*lengths[-1],lengths,points[:,axis]) for axis in (0,1)])
        chord=points[0]*(1-fraction[:,None])+points[-1]*fraction[:,None]
        expected=((pts-chord)/1000).ravel();errors.append(float(np.max(np.abs(expected-out[i,:14]))))
    assert max(errors)<1e-5,(kind,max(errors));checks[kind]=dict(independent_routes=len(selected),max_shape_error_km=max(errors))
write_json(BASE/'instrumentos/geometria_rutas.json',dict(passed=True,checks=checks,test_labels_read=False,method='Independent Python predecessor trace plus NumPy polyline interpolation, compared to parallel Numba implementation. Includes partial chains and same-chain cases.'))
print(checks,flush=True)
