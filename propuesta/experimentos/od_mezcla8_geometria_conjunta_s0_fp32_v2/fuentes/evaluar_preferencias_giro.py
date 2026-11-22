from common import *
from preferencias_giro import angle_bin
from map_matching import node_connections,node_distance
from numba import njit,set_num_threads

@njit(cache=True)
def evaluate(events,local,shared,outptr,outchains,offset,heading_in,heading_out,chain_dst,junction_raw,chain_length,D,conn):
    n=len(events);correct=np.zeros((n,3),np.float64);logloss=np.zeros((n,3),np.float64);support=np.zeros(n);valid=np.zeros(n,np.bool_);choices=np.full((n,3),-1,np.int32)
    for i in range(n):
        incoming,actual,ego,dist,destination,trip,fold=events[i];node=chain_dst[incoming];lo=outptr[node];hi=outptr[node+1];k=hi-lo
        cost=np.empty(k);turn=np.empty(k,np.int32);counts=np.zeros(k);prior=np.zeros(k);actual_slot=-1
        for s in range(k):
            c=outchains[lo+s];cost[s]=chain_length[c]+node_distance(junction_raw[chain_dst[c]],destination,D,*conn);turn[s]=angle_bin(heading_in[incoming,0],heading_in[incoming,1],heading_out[c,0],heading_out[c,1])
            for f in range(5):
                if f!=fold:counts[s]+=local[f,ego,dist,offset[incoming]+s]
            if c==actual:actual_slot=s
        if actual_slot<0 or not np.isfinite(cost[actual_slot]):continue
        shortest=np.min(cost)
        if not np.isfinite(shortest):continue
        shared_counts=np.ones(8)
        for f in range(5):
            if f!=fold:shared_counts+=shared[f,ego,dist]
        multiplicity=np.bincount(turn,minlength=8)
        physical=np.exp(-np.minimum((cost-shortest)/500,700));physical=physical/max(physical.sum(),1e-30)
        for s in range(k):prior[s]=physical[s]*shared_counts[turn[s]]/multiplicity[turn[s]]
        prior=prior/max(prior.sum(),1e-30);support[i]=counts.sum();posterior=(counts+20*prior)/(counts.sum()+20)
        choices[i,0]=outchains[lo+np.argmin(cost)];choices[i,1]=outchains[lo+np.argmax(prior)];choices[i,2]=outchains[lo+np.argmax(posterior)]
        for m in range(3):correct[i,m]=choices[i,m]==actual
        logloss[i,0]=-np.log(max(physical[actual_slot],1e-12));logloss[i,1]=-np.log(max(prior[actual_slot],1e-12));logloss[i,2]=-np.log(max(posterior[actual_slot],1e-12));valid[i]=True
    return correct,logloss,support,valid,choices

if __name__=='__main__':
    started=time.time();set_num_threads(4);folder=DATA/'preferencias_giro_2014';manifest=json.loads((folder/'manifiesto.json').read_text());z=dict(np.load(folder/'muestra_oof.npz'));structure=dict(np.load(folder/'estructura.npz'));comp=dict(np.load(DATA/'grafo_compacto_2014.npz'));D=np.load(DATA/'distancias_2014/D.npy',mmap_mode='r');conn=node_connections(comp);local=np.load(folder/'local.npy',mmap_mode='r');shared=np.load(folder/'global.npy');e=z['events']
    args=(structure['outptr'],structure['outchains'],structure['transition_offsets'],structure['heading_in'],structure['heading_out'],comp['chain_dst'],comp['junction_raw'],comp['chain_length'],D,conn)
    correct,ll,support,valid,choices=evaluate(e,local,shared,*args)
    # Intervene on a nonzero own-fold context. The held-out prediction must not change.
    probe=e[:30];modified=np.array(local,copy=True);modified_shared=shared.copy();f=int(probe[0,6]);modified[f]+=100000;modified_shared[f]+=100000
    probe=probe[probe[:,6]==f];old=evaluate(probe,local,shared,*args);new=evaluate(probe,modified,modified_shared,*args)
    assert all(np.array_equal(a,b) for a,b in zip(old,new));assert len(probe)>0
    dates=z['trip_timestamp'][e[valid,5]]//86400;_,inv=np.unique(dates,return_inverse=True);count=np.bincount(inv);delta=correct[valid,2]-correct[valid,0];sums=np.bincount(inv,weights=delta);ix=np.random.default_rng(1729).integers(0,len(count),(4000,len(count)));boot=sums[ix].sum(1)/count[ix].sum(1)
    output=dict(task='Teacher-forced next-chain prediction on sampled training decisions, whole-trip-week out of fold. Not OD travel-time evaluation.',source_manifest_sha256=sha(folder/'manifiesto.json'),n=len(e),eligible_n=int(valid.sum()),unreachable_or_invalid_n=int((~valid).sum()),methods=['minimum_network_length','shared_turn_and_network_prior','local_counts_with_dirichlet20_prior'],accuracy=correct[valid].mean(0),cross_entropy=ll[valid].mean(0),local_support_quantiles=np.quantile(support[valid],[0,.1,.5,.9,1]),paired_accuracy_gain=float(delta.mean()),paired_day_bootstrap95=np.quantile(boot,[.025,.975]),own_fold_intervention_passed=True,plan_decoder=bool(delta.mean()>=.02),decision_rule='Implement decoder only if local OOF accuracy exceeds shortest-network decision by >=2 percentage points. Preliminary engineering gate, not a statistical selection claim.',test_labels_read=False,seconds=time.time()-started)
    write_json(BASE/'instrumentos/preferencias_giro_oof.json',output);np.savez_compressed(BASE/'instrumentos/preferencias_giro_oof.npz',events=e,valid=valid,correct=correct,logloss=ll,support=support,choices=choices);print(json.dumps(output,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)),flush=True)
