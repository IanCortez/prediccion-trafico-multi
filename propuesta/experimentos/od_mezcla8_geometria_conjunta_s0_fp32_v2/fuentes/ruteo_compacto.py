from common import *
from features import ORIGIN
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
def compressed_csr(comp,kind):
    weight=comp["chain_length"] if kind=="length" else comp["chain_free"]
    n=len(comp["junction_raw"]);pair=comp["chain_src"].astype(np.int64)*n+comp["chain_dst"]
    order=np.lexsort((weight,pair));sorted_pair=pair[order]
    chosen=order[np.r_[True,np.diff(sorted_pair)!=0]]
    # chosen is sorted by (source,destination), exactly CSR storage order.
    graph=csr_matrix((weight[chosen],(comp["chain_src"][chosen],comp["chain_dst"][chosen])),shape=(n,n))
    return graph,chosen
def query_map(d,raw,comp):
    active=np.unique(np.r_[raw["u"],raw["v"]]);tree=cKDTree(raw["xy"][active])
    def metric(tag):
        return np.column_stack([(d[tag+"_lon"]-ORIGIN[0])*111195*np.cos(np.deg2rad(ORIGIN[1])),(d[tag+"_lat"]-ORIGIN[1])*111195])
    q={}
    for tag in ("o","d"):
        snap,index=tree.query(metric(tag));node=active[index];j=comp["to_junction"][node]
        chain=comp["node_chain"][node].copy();pos=comp["node_pos"][node].copy()
        js=np.full(chain.shape,-1,dtype=np.int32);leng=np.full(chain.shape,np.inf);free=leng.copy()
        for slot in range(2):
            valid=chain[:,slot]>=0;c=chain[valid,slot];p=pos[valid,slot]
            if tag=="o":
                js[valid,slot]=comp["chain_dst"][c]
                leng[valid,slot]=comp["chain_length"][c]-comp["prefix_length"][p]
                free[valid,slot]=comp["chain_free"][c]-comp["prefix_free"][p]
            else:
                js[valid,slot]=comp["chain_src"][c]
                leng[valid,slot]=comp["prefix_length"][p];free[valid,slot]=comp["prefix_free"][p]
        junction=j>=0;js[junction,0]=j[junction];leng[junction,0]=0;free[junction,0]=0
        js[junction,1]=-1;leng[junction,1]=np.inf;free[junction,1]=np.inf
        chain[junction]=-1;pos[junction]=-1
        q[tag+"_junction"]=js;q[tag+"_chain"]=chain;q[tag+"_pos"]=pos;q[tag+"_length"]=leng;q[tag+"_free"]=free;q[tag+"_snap"]=snap.astype(np.float32)
    return q
def choose_static(q,dist,source_lookup,comp,kind):
    n=len(q["o_snap"]);best=np.full(n,np.inf);choice=np.full(n,-1,dtype=np.int8)
    metric="length" if kind=="length" else "free"
    prefix=comp["prefix_"+metric]
    for o in range(2):
        for d in range(2):
            source=q["o_junction"][:,o];dest=q["d_junction"][:,d]
            valid=(source>=0)&(dest>=0)
            ix=np.flatnonzero(valid);rows=source_lookup[source[ix]]
            assert (rows>=0).all()
            cand=dist[rows,dest[ix]].astype(np.float64)+q["o_"+metric][ix,o]+q["d_"+metric][ix,d]
            good=cand<best[ix];use=ix[good];best[use]=cand[good];choice[use]=o*2+d
            same=(q["o_chain"][:,o]>=0)&(q["o_chain"][:,o]==q["d_chain"][:,d])&(q["d_pos"][:,d]>=q["o_pos"][:,o])
            ix=np.flatnonzero(same)
            cand=prefix[q["d_pos"][ix,d]]-prefix[q["o_pos"][ix,o]]
            good=cand<best[ix];use=ix[good];best[use]=cand[good];choice[use]=4+o*2+d
    return best,choice

