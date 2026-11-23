"""Implicit ordered historical subtrip retrieval; no actual query path/N/time target."""
from base_replanteo import *
from scipy.spatial import cKDTree
from numba import njit,prange,set_num_threads
from datos_progreso_recurrente import lisbon_transitions
@njit(cache=True)
def clock_vars(ts,starts,ends):
    offset=0
    for k in range(len(starts)):
        if starts[k]<=ts<ends[k]:offset=3600
    local=ts+offset
    return (local%86400)/3600.,((local//86400+3)%7)>=5

@njit(cache=True,parallel=True)
def choose_subtrips(oids,dids,odist,ddist,parent_of_point,offsets,parent_n,parent_ts,parent_fold,xy,
                    query_xy,query_ts,query_fold,starts,ends):
    B,K=oids.shape;parent=np.full((B,64),-1,np.int32)
    first=np.zeros((B,64),np.uint8);last=np.zeros((B,64),np.uint8)
    duration=np.zeros((B,64),np.float32);logweight=np.full((B,64),-1e30,np.float64)
    error=np.full((B,64),np.inf,np.float32);is_full=np.zeros((B,64),np.bool_)
    for q in prange(B):
        op=parent_of_point[oids[q]];dp=parent_of_point[dids[q]]
        so=np.argsort(op);sd=np.argsort(dp);a=0;b=0
        qhour,qweekend=clock_vars(query_ts[q],starts,ends)
        qlen=np.sqrt((query_xy[q,2]-query_xy[q,0])**2+(query_xy[q,3]-query_xy[q,1])**2)
        while a<K and b<K:
            po=op[so[a]];pd=dp[sd[b]]
            if po<pd:
                a+=1;continue
            if pd<po:
                b+=1;continue
            ae=a+1;be=b+1
            while ae<K and op[so[ae]]==po:ae+=1
            while be<K and dp[sd[be]]==po:be+=1
            if query_fold[q]<5 and parent_fold[po]==query_fold[q]:
                a=ae;b=be;continue
            best=1e30;bi=-1;bj=-1;blen=0.
            for ia in range(a,ae):
                ao=so[ia];oi=oids[q,ao];i=oi-offsets[po]
                for ib in range(b,be):
                    ad=sd[ib];di=dids[q,ad];j=di-offsets[po]
                    if j-i<4:continue
                    seconds=15*(j-i)
                    dx=xy[di,0]-xy[oi,0];dy=xy[di,1]-xy[oi,1];length=np.sqrt(dx*dx+dy*dy)
                    if length<200 or length/seconds*3.6>120:continue
                    cost=odist[q,ao]**2+ddist[q,ad]**2
                    # Stable geometric ties: earlieststart/latestend; never queryT based.
                    if cost<best-1e-10 or (abs(cost-best)<=1e-10 and (i<bi or (i==bi and j>bj))):
                        best=cost;bi=i;bj=j;blen=length
            if bi>=0:
                hour,weekend=clock_vars(parent_ts[po]+15*bi,starts,ends)
                delta=(hour-qhour+12)%24-12
                prior=-.5*best/(50*50)-.5*(delta/4)**2+(0. if weekend==qweekend else np.log(.35))
                slot=0
                for k in range(1,64):
                    if logweight[q,k]<logweight[q,slot]:slot=k
                if prior>logweight[q,slot]:
                    parent[q,slot]=po;first[q,slot]=bi;last[q,slot]=bj
                    scale=min(2.,max(.5,qlen/max(blen,1.)))
                    duration[q,slot]=15*(bj-bi)*scale
                    logweight[q,slot]=prior;error[q,slot]=np.sqrt(best)
                    is_full[q,slot]=(bi==0 and bj==parent_n[po]-1)
            a=ae;b=be
    return parent,first,last,duration,logweight,error,is_full

def metric_xy(d):
    sx=111195*np.cos(np.deg2rad(41.16))
    return np.column_stack([(d['o_lon']+8.62)*sx,(d['o_lat']-41.16)*111195,
                            (d['d_lon']+8.62)*sx,(d['d_lat']-41.16)*111195])

class SubtrajectoryMemory:
    def __init__(self):
        folder=OUT/'memoria_puntos_Train_s0';meta=read(folder/'resultado.json')
        assert meta['complete'] and sha(folder/'puntos_Train.npz')==meta['outputs_sha256']
        self.data=dict(np.load(folder/'puntos_Train.npz'));z=self.data
        self.parent=np.repeat(np.arange(len(z['parent_gps_n']),dtype=np.int32),z['parent_gps_n'])
        assert len(self.parent)==len(z['xy_m'])
        self.indexable=np.unpackbits(z['indexable_packed'],count=len(self.parent)).astype(bool)
        self.tree=None;self.tree_ids=None;self.tree_fold=None
        self.starts,self.ends=lisbon_transitions();set_num_threads(2)
    def build_tree(self,fold=5):
        assert 0<=fold<=5;z=self.data
        selected=self.indexable.copy()
        if fold<5:selected&=z['parent_fold'][self.parent]!=fold
        ids=np.flatnonzero(selected)
        if fold<5:assert np.all(z['parent_fold'][self.parent[ids]]!=fold)
        self.tree=None;self.tree_ids=None
        self.tree_ids=ids;self.tree=cKDTree(z['xy_m'][ids].astype(np.float64));self.tree_fold=fold
    def query(self,d,training=False,k=2048):
        from consulta_od import od_records
        assert set(d)=={'o_lon','o_lat','d_lon','d_lat','timestamp'}
        q=od_records([{key:int(d[key][i]) if key=='timestamp' else float(d[key][i]) for key in d} for i in range(len(d['timestamp']))])
        xy=metric_xy(q);fold=(q['timestamp']//604800)%5 if training else np.full(len(xy),5)
        assert len(np.unique(fold))==1 and self.tree_fold==int(fold[0])
        if not training:
            z=self.data;have=z['eligible']
            assert np.max(z['timestamp'][have]+z['time_s'][have])<q['timestamp'].min()
        od,oi=self.tree.query(xy[:,:2],k=k,workers=2);dd,di=self.tree.query(xy[:,2:],k=k,workers=2)
        z=self.data
        return choose_subtrips(self.tree_ids[oi],self.tree_ids[di],od,dd,self.parent,z['offsets'],z['parent_gps_n'],
            z['timestamp'],z['parent_fold'],z['xy_m'],xy,q['timestamp'],fold,self.starts,self.ends)
