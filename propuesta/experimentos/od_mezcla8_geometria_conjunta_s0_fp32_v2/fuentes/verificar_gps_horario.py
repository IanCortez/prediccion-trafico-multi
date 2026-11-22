from common import *
from ruteo_compacto import *
import ast
from numba import njit,prange
raw=dict(np.load(DATA/'mapa_dirigido.npz'));comp=dict(np.load(DATA/'grafo_compacto.npz'));q=dict(np.load(DATA/'consultas_ruteo_compacto.npz'));arc=dict(np.load(DATA/'arcos_gps_horario.npz'));profile=arc['query_profile']
assert np.all(profile[q['split_bounds'][1]:]>=336)
errors={}
for kind,start in (('length',0),('free',7)):
    X=np.concatenate([np.load(DATA/(s+'_gps_horario.npz'))['X'] for s in ('train','validation','test')]);graph,chosen=compressed_csr(comp,kind);pred=np.load(DATA/f'compact_{kind}_pred.npy',mmap_mode='r');choice=np.load(DATA/f'compact_{kind}_choices.npz')['choice'];differences=[]
    for i in np.random.default_rng(815).choice(np.flatnonzero(X[:,start+6]>0),40,replace=False):
        c=int(choice[i]);os=(c%4)//2;ds=c%2;arcs=[]
        def add(chain,lo=None,hi=None):
            if lo is None:lo=comp['chain_offsets'][chain]
            if hi is None:hi=comp['chain_offsets'][chain+1]-1
            arcs.extend(comp['flat_incoming_arc'][lo+1:hi+1].tolist())
        if c>=4:add(q['o_chain'][i,os],q['o_pos'][i,os],q['d_pos'][i,ds])
        else:
            chain=q['o_chain'][i,os]
            if chain>=0:add(chain,q['o_pos'][i,os])
            root=q['o_junction'][i,os];node=q['d_junction'][i,ds];sr=q['source_lookup'][root]
            while node!=root:
                parent=pred[sr,node];edge=graph.indptr[parent]+np.flatnonzero(graph.indices[graph.indptr[parent]:graph.indptr[parent+1]]==node)[0];add(chosen[edge]);node=parent
            chain=q['d_chain'][i,ds]
            if chain>=0:add(chain,hi=q['d_pos'][i,ds])
        expected=arc['time_s'][profile[i],arcs].astype(float).sum()+(q['o_snap'][i]+q['d_snap'][i])/(20/3.6);differences.append(abs(expected-X[i,start]*600))
    assert max(differences)<.1;errors[kind]=max(differences)
# Extract only the trusted accumulator function, avoiding preprocessing side effects.
tree=ast.parse((BASE/'preparar_estado_gps.py').read_text(encoding='utf-8-sig'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='summarize');fn.decorator_list=[];env=dict(np=np,njit=njit,prange=prange);exec(compile(ast.Module(body=[fn],type_ignores=[]),'summarize_contract','exec'),env)
xy=np.array([[-.5,.5],[-.48,.5],[-.5,.5],[-.44,.5]],np.float32);offsets=np.array([0,2,4]);ts=np.array([1372670000,1372670000]);fold=np.array([0,1]);transitions=np.array([1372630000]);offset=np.array([3600])
D,S,_=env['summarize'](xy,offsets,ts,fold,transitions,offset);mutated=xy.copy();mutated[1,0]=-.46;D2,S2,_=env['summarize'](mutated,offsets,ts,fold,transitions,offset)
assert np.allclose(D.sum(0)-D[0],D2.sum(0)-D2[0]) and np.array_equal(S.sum(0)-S[0],S2.sum(0)-S2[0]);assert not np.allclose(D.sum(0),D2.sum(0))
write_json(BASE/'instrumentos/gps_horario.json',dict(passed=True,independent_raw_arc_queries=80,max_cost_error_s=errors,heldout_uses_training_full_profile=True,query_fold_gps_intervention_invariant=True,full_profile_changes_under_intervention=True,test_labels_read=False))
print(dict(errors=errors,fold_intervention_passed=True),flush=True)

