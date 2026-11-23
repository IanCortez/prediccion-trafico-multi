"""Read-only corrected Train teacher sequences; no duplicated token cache."""
from base_replanteo import *
import torch
from datetime import datetime, timezone, timedelta

def lisbon_transitions():
    # EU public calendar rule, coverage enclosing all Porto queries and rollouts.
    starts=[]; ends=[]
    for year in range(2012,2017):
        for month,output in [(3,starts),(10,ends)]:
            day=datetime(year,month+1,1,tzinfo=timezone.utc)-timedelta(days=1)
            day-=timedelta(days=(day.weekday()-6)%7)
            output.append(int(day.replace(hour=1).timestamp()))
    return np.asarray(starts),np.asarray(ends)

def load_graph():
    path=OUT/'banco_valor_vial_s0/grafo_estatico.npz'
    g=dict(np.load(path)); g['D']=np.load(DATA/'distancias_2014/D.npy',mmap_mode='r')
    g['summer_start'],g['summer_end']=lisbon_transitions()
    return g

class CorrectedSequences:
    def __init__(self, max_fragment=128):
        self.max_fragment=max_fragment
        core=OUT/'datos_calles_parciales_Train_s0'; adapter=OUT/'adaptador_progreso_vial_s0'
        cm=read(core/'resultado.json'); am=read(adapter/'resultado.json')
        proof=read(OUT/'verificacion_adaptador_progreso_vial.json')
        assert proof['passed'] and proof['adapter_manifest_sha256']==sha(adapter/'resultado.json')
        n=cm['tokens']; nt=1023529
        self.code=np.empty(n,np.uint16);self.entry=np.empty(n,np.float32)
        self.distance=np.empty(n,np.float32);self.dt=np.empty(n,np.float32)
        self.elapsed=np.empty(n,np.float32);self.remaining=np.empty(n,np.float32)
        self.route=np.full(n,-1,np.int8)
        self.destination=np.full(nt,53766,np.int32);self.destpos=np.zeros(nt,np.float32)
        self.summary=np.zeros((nt,18),np.float32)
        g=dict(np.load(OUT/'banco_valor_vial_s0/grafo_estatico.npz'))
        lengths=g['chain_length']; static=g['static']; nbr=g['neighbors']
        fragments=[]; cursor=0
        for info,ai in zip(cm['shards'],am['files']):
            p=core/info['file'];a=adapter/ai['file']
            assert sha(p)==info['sha256'] and sha(a)==ai['sha256']
            z=dict(np.load(p));q=dict(np.load(a));m=len(z['chain'])
            code=z['chain'].astype(np.int32);known=np.unpackbits(q['position_known_packed'],count=m).astype(bool)
            reset=np.unpackbits(q['reset_before_packed'],count=m).astype(bool)
            macro=np.unpackbits(q['macro_next_valid_packed'],count=m).astype(bool)
            entry=np.zeros(m,np.float32);entry[q['exception_index']]=q['entry_m']
            self.code[cursor:cursor+m]=np.where(code==65535,53766,code).astype(np.uint16)
            self.entry[cursor:cursor+m]=entry
            self.distance[cursor:cursor+m]=z['distance_m'];self.dt[cursor:cursor+m]=z['duration_s']
            ii=np.flatnonzero(macro);nxt=code[ii+1]
            matches=nbr[code[ii]]==nxt[:,None];assert matches.any(1).all()
            self.route[cursor+ii]=matches.argmax(1).astype(np.int8)
            for j,(a0,b0) in enumerate(zip(z['offsets'][:-1],z['offsets'][1:])):
                ti=int(z['train_index'][j]);dc=int(z['terminal_chain'][j]);target=float(z['true_duration_s'][j])
                dt=z['duration_s'][a0:b0].astype(np.float64);elapsed=np.r_[0.,np.cumsum(dt)[:-1]]
                assert abs(dt.sum()-target)<.001
                self.elapsed[cursor+a0:cursor+b0]=elapsed;self.remaining[cursor+a0:cursor+b0]=target-elapsed
                if dc>=53766: continue
                self.destination[ti]=dc;pos=float(z['terminal_offset_m'][j])
                if g['chain_src'][dc]==g['chain_dst'][dc] and pos<=1e-4:pos=float(lengths[dc])
                self.destpos[ti]=pos
                inds=np.arange(a0,b0)[known[a0:b0]]
                if len(inds):
                    roads=code[inds];ds=z['distance_m'][inds].astype(np.float64)
                    avg=static[roads].mean(0)
                    totals=[np.log1p(target)/8,np.log1p(ds.sum())/10,
                            np.log1p(len(dt))/6,np.log1p(dt.std())/6,
                            np.log1p(np.sum(ds/np.maximum(static[roads,10]*20,.1)))/8,
                            float(known[a0:b0].mean())]
                    self.summary[ti]=np.r_[avg,totals]
                if q['terminal_eos_valid'][j]:self.route[cursor+b0-1]=6
                a1=int(a0)
                while a1<b0:
                    if not known[a1]:a1+=1;continue
                    b1=a1+1
                    while b1<b0 and known[b1] and not reset[b1] and b1-a1<max_fragment:b1+=1
                    fragments.append((cursor+a1,b1-a1,ti));a1=b1
            cursor+=m
        assert cursor==n
        self.fragments=np.asarray(fragments,np.int64)
        self.index=dict(np.load(OUT/'indices_progreso_vial_s0/indices_Train.npz'))
        self.eligible=self.index['eligible_primary_train_index']
        self.y=np.load(DATA/'train_y.npy');self.ts=np.load(DATA/'train_x.npz')['timestamp']
        assert len(self.eligible)==1023485 and self.fragments[:,1].max()<=max_fragment
        self.bytes=sum(v.nbytes for v in vars(self).values() if isinstance(v,np.ndarray))
    def batch(self, fragment_indices, device='cuda'):
        a=self.fragments[fragment_indices];length=a[:,1];B=len(a);L=int(length.max())
        k=np.arange(L)[None,:];valid=k<length[:,None];ii=np.minimum(a[:,0,None]+k,a[:,0,None]+length[:,None]-1)
        ti=a[:,2];mask=valid
        arrays=dict(chain=self.code[ii].astype(np.int64),entry_m=self.entry[ii],
            distance_m=self.distance[ii],local_s=self.dt[ii],elapsed_s=self.elapsed[ii],
            remaining_s=self.remaining[ii],route_target=self.route[ii].astype(np.int64),
            valid=mask,destination=self.destination[ti].astype(np.int64),destination_m=self.destpos[ti],
            recognition=self.summary[ti],timestamp=self.ts[ti],length=length,
            attachment=self.index['attachment_pair'][ti].astype(np.int64),train_index=ti)
        return {key:torch.as_tensor(value,device=device) for key,value in arrays.items()}
