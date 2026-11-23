"""Reconstruct native15s Train cells only in RAM; no query length at inference."""
import numpy as np,hashlib,struct
from modelo_direccional import heading_ids
import torch

def query_seeds(data):
    columns=[np.asarray(data[k]) for k in ['o_lon','o_lat','d_lon','d_lat','timestamp']]
    return np.fromiter((int.from_bytes(hashlib.blake2s(struct.pack('<ddddq',float(a),float(b),float(c),float(d),int(t)),digest_size=4).digest(),'little') for a,b,c,d,t in zip(*columns)),dtype=np.int64,count=len(columns[0]))

class NativeTrainSequences:
    def __init__(self,path,neighbors,allowed,ny,N):
        r=np.load(path)
        self.cell=r['cell'].astype(np.int32)
        self.duration=r['duration_n'].astype(np.int64)
        self.offset=r['run_offsets']
        self.trip=r['trip_train_index'].astype(np.int64)
        self.neighbors=neighbors
        self.allowed=allowed
        self.ny=ny
        self.N=N

    def make(self,indices,origin,destination,y,horizon=120):
        B=len(indices)
        cell=np.empty((B,horizon),np.int64)
        heading=np.full((B,horizon),4,np.int64)
        age=np.zeros((B,horizon),np.int64)
        action=np.zeros((B,horizon),np.int64)
        event_mask=np.zeros((B,horizon),bool)
        motion_mask=np.zeros_like(event_mask)
        terminal=np.zeros_like(event_mask)
        rows=self.trip[np.asarray(indices)]
        for b,ti in enumerate(indices):
            first,last=self.offset[ti:ti+2]
            cc=self.cell[first:last]
            dd=self.duration[first:last].copy()
            dd[-1]+=1
            point=np.repeat(cc,dd)
            L=len(point)
            assert 5<=L<=horizon and (L-1)*15==y[rows[b]]
            assert point[0]==origin[rows[b]] and point[-1]==destination[rows[b]]
            cell[b]=origin[rows[b]]
            cell[b,:L]=np.where(point<self.N,point,origin[rows[b]])
            good=point<self.N
            event_mask[b,:L]=good
            terminal[b,L-1]=True
            run_heading=np.full(len(cc),4,np.int64)
            if len(cc)>1:
                dx=cc[1:]//self.ny-cc[:-1]//self.ny
                dy=cc[1:]%self.ny-cc[:-1]%self.ny
                h=np.where(np.abs(dx)>=np.abs(dy),np.where(dx>=0,0,1),np.where(dy>=0,2,3))
                known=(cc[1:]<self.N)&(cc[:-1]<self.N)
                run_heading[1:]=np.where(known,h,4)
            heading[b,:L]=np.repeat(run_heading,dd)
            age[b,:L]=np.concatenate([np.arange(n) for n in dd])
            dx=point[1:]//self.ny-point[:-1]//self.ny
            dy=point[1:]%self.ny-point[:-1]%self.ny
            same=point[1:]==point[:-1]
            code=(dx+3)*7+dy+3
            aa=code-(code>24)
            small=(np.abs(dx)<=3)&(np.abs(dy)<=3)&(~same)
            safe=np.clip(aa,0,47)
            motion_mask[b,:L-1]=good[:-1]&good[1:]&(same|(small&self.allowed[cell[b,:L-1],safe]))
            action[b,:L-1]=np.where(same,0,safe+1)
        return rows,dict(cell=cell,heading=heading,age=age,action=action,event_mask=event_mask,motion_mask=motion_mask,terminal=terminal)

def prefix_loss(motion,stop,batch):
    eps=1e-7
    event=-torch.where(batch['terminal'],stop.clamp_min(eps).log(),torch.log1p(-stop.clamp_max(1-eps)))
    route=-motion.gather(-1,batch['action'][:,:,None]).squeeze(-1).clamp_min(eps).log()
    n=120  # Constant scale: dividing by observed length would change the fitted law.
    total=((event*batch['event_mask']).sum(-1)+(route*batch['motion_mask']).sum(-1))/n
    return total.mean()
