import torch,math
from torch import nn
class TrajectoryEncoder(nn.Module):
    """Masked trajectory representation; OD inference receives exactly two points.
    Original point indices and path length are never input features.
    """
    def __init__(self,context_grid,grid_info,coef,dim=128,layers=2,dropout=.1):
        super().__init__()
        self.register_buffer("context_grid",torch.as_tensor(context_grid,dtype=torch.float32))
        self.register_buffer("coef",torch.as_tensor(coef,dtype=torch.float32))
        self.grid_info=grid_info
        self.periods=(.25,.5,1.,2.,4.,8.)
        n=4+8*len(self.periods)+context_grid.shape[1]+1+6+3
        self.point=nn.Sequential(nn.Linear(n,dim),nn.GELU(),nn.Linear(dim,dim),nn.LayerNorm(dim))
        layer=nn.TransformerEncoderLayer(dim,4,dim*4,dropout,batch_first=True,norm_first=True,activation="gelu")
        self.encoder=nn.TransformerEncoder(layer,layers,norm=nn.LayerNorm(dim),enable_nested_tensor=False)
        self.time_head=nn.Sequential(nn.Linear(dim,64),nn.GELU(),nn.Linear(64,1))
        self.spatial_head=nn.Linear(dim,2)
        nn.init.zeros_(self.time_head[-1].weight);nn.init.zeros_(self.time_head[-1].bias)
    def context(self,xy):
        g=self.grid_info;x=((xy[...,0]*1000-g["x0"])/g["gridstep"]).floor().long()
        y=((xy[...,1]*1000-g["y0"])/g["gridstep"]).floor().long()
        valid=(x>=0)&(x<g["nx"])&(y>=0)&(y<g["ny"])
        idx=x.clamp(0,g["nx"]-1)*g["ny"]+y.clamp(0,g["ny"]-1)
        return torch.cat([self.context_grid[idx]*valid[...,None]/5,valid[...,None].float()],-1)
    def forward(self,xy,known_seconds,visible,departure,pad=None,spatial_visible=None,return_hidden=False,task_id=None):
        rel=xy-xy[:,:1];geo=[xy/5,rel/5]
        for period in self.periods:
            for loc in (xy,rel):
                angle=loc*(2*math.pi/period);geo.extend([angle.sin(),angle.cos()])
        geo.append(self.context(xy));geo=torch.cat(geo,-1)
        if spatial_visible is not None:geo=geo*spatial_visible[...,None]
        role=torch.zeros_like(known_seconds);role[:,0]=1
        temporal=torch.stack([known_seconds*visible/600,visible.float(),role],-1)
        dep=departure[:,None,:].expand(-1,xy.shape[1],-1)
        embedded=self.point(torch.cat([geo,temporal,dep],-1))
        if hasattr(self,"task_prompt"):
            if task_id is None:task_id=torch.zeros(len(xy),dtype=torch.long,device=xy.device)
            embedded=embedded+self.task_prompt(task_id)[:,None,:]
        h=self.encoder(embedded,src_key_padding_mask=pad)
        correction=3*torch.tanh(self.time_head(h).squeeze(-1)/3)
        distance=(rel.square().sum(-1).sqrt()*1000).clamp_min(20)
        baseline=self.coef[0]+self.coef[1]*distance.log()
        seconds=(baseline+correction).float().exp()
        return (seconds,self.spatial_head(h),h) if return_hidden else (seconds,self.spatial_head(h))
def load_grid(path):
    import numpy as np
    z=np.load(path);return z["features"],{k:int(z[k]) for k in ("gridstep","x0","y0","nx","ny")}


def warm_load(model,checkpoint):
    incoming=checkpoint["state"];current=model.state_dict();expanded=[]
    for key,value in incoming.items():
        if key=="context_grid":continue
        if key=="point.0.weight" and value.shape!=current[key].shape:
            dst=torch.zeros_like(current[key]);prefix=4+8*len(model.periods)+54
            dst[:,:prefix]=value[:,:prefix];dst[:,-10:]=value[:,-10:]
            current[key]=dst;expanded.append(dict(key=key,old=list(value.shape),new=list(dst.shape),new_features_initialized_zero=True))
        else:
            assert key in current and current[key].shape==value.shape,(key,value.shape)
            current[key]=value
    model.load_state_dict(current)
    return expanded



class DurationClassifier(TrajectoryEncoder):
    """Ordered duration distribution; its median is the MAE-optimal point statistic."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.register_buffer("duration_grid",torch.arange(60,1785+1,15,dtype=torch.float32))
        self.duration_head=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,116))
        nn.init.zeros_(self.duration_head[-1].weight);nn.init.zeros_(self.duration_head[-1].bias)
    def posterior(self,xy,departure):
        assert xy.shape[1]==2,"OD distribution accepts exactly two endpoints"
        known=torch.zeros(xy.shape[:2],device=xy.device);visible=torch.zeros_like(known,dtype=torch.bool);visible[:,0]=True
        _,_,h=super().forward(xy,known,visible,departure,return_hidden=True)
        distance=((xy[:,1]-xy[:,0]).square().sum(-1).sqrt()*1000).clamp_min(20)
        baseline=self.coef[0]+self.coef[1]*distance.log()
        logtime=self.duration_grid.log()
        prior=-.5*((logtime[None,:]-baseline[:,None])/.55)**2-logtime[None,:]
        return self.duration_head(h[:,1]).float()+prior
    def quantiles(self,logits,levels=(.1,.5,.9)):
        cdf=logits.softmax(-1).cumsum(-1)
        return torch.stack([self.duration_grid[(cdf<q).sum(-1).clamp_max(115)] for q in levels],-1)


class RouteAuxEncoder(TrajectoryEncoder):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        state=torch.get_rng_state()
        self.route_head=nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Linear(128,15))
        nn.init.zeros_(self.route_head[-1].weight);nn.init.zeros_(self.route_head[-1].bias)
        torch.set_rng_state(state)



class TaskAwareEncoder(TrajectoryEncoder):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        state=torch.get_rng_state()
        self.task_prompt=nn.Embedding(2,128);nn.init.zeros_(self.task_prompt.weight)
        torch.set_rng_state(state)

