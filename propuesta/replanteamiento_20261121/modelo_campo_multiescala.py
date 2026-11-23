"""OD-only multiresolution bilinear endpoint fields plus route-duration mixture."""
import torch,math
from torch import nn
from modelo_mezcla_conjunta import JointRouteTimeMixture
class BilinearEndpointLevel(nn.Module):
    def __init__(self,step,width=8):
        super().__init__();self.step=float(step);self.nx=int(math.ceil(13000/step))+1;self.ny=int(math.ceil(7000/step))+1
        self.embedding=nn.Embedding(self.nx*self.ny,width);nn.init.normal_(self.embedding.weight,0,.02)
        self.register_buffer('confidence',torch.ones(self.nx*self.ny))
    def indices_weights(self,xy_m):
        pos=(xy_m-xy_m.new_tensor([-6500.,-3000.]))/self.step
        ix=pos[:,0].floor().long().clamp(0,self.nx-2);iy=pos[:,1].floor().long().clamp(0,self.ny-2)
        fx=(pos[:,0]-ix).clamp(0,1);fy=(pos[:,1]-iy).clamp(0,1)
        ids=torch.stack([ix*self.ny+iy,(ix+1)*self.ny+iy,ix*self.ny+iy+1,(ix+1)*self.ny+iy+1],-1)
        weights=torch.stack([(1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy],-1)
        return ids,weights
    def forward(self,xy_m):
        ids,w=self.indices_weights(xy_m)
        return (self.embedding(ids)*self.confidence[ids][...,None]*w[...,None]).sum(1)
    @torch.no_grad()
    def fit_confidence(self,xy_m,shrinkage=20.):
        assert self.embedding.weight.device.type=='cpu'
        counts=torch.zeros(self.nx*self.ny,dtype=torch.float64)
        for block in xy_m.split(200000):
            ids,w=self.indices_weights(block)
            counts+=torch.bincount(ids.reshape(-1),weights=w.reshape(-1).double(),minlength=len(counts))
        self.confidence.copy_((counts/(counts+shrinkage)).float())
        return dict(nodes=len(counts),supported_nodes=int((counts>0).sum()),mean_confidence=float(self.confidence.mean()),total_endpoint_weight=float(counts.sum()))
class MultiscaleODField(nn.Module):
    def __init__(self,clock_indices,steps=(1000.,500.,250.,125.,62.5,31.25),width=8):
        super().__init__();self.steps=tuple(steps);self.width=width
        self.origin=nn.ModuleList([BilinearEndpointLevel(s,width) for s in steps]);self.destination=nn.ModuleList([BilinearEndpointLevel(s,width) for s in steps])
        self.register_buffer('clock_indices',torch.tensor(clock_indices,dtype=torch.long))
        self.net=nn.Sequential(nn.Linear(len(steps)*width*3+len(clock_indices),128),nn.SiLU(),nn.Linear(128,248))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):
        a=torch.cat([level(x[:,:2]*1000) for level in self.origin],-1)
        b=torch.cat([level(x[:,2:4]*1000) for level in self.destination],-1)
        return self.net(torch.cat([a,b,a*b,x[:,self.clock_indices]],-1))
    @torch.no_grad()
    def fit_confidence(self,coordinates_km):
        assert coordinates_km.shape[1]==4 and torch.isfinite(coordinates_km).all()
        results=[]
        for side,levels,coords in [('o',self.origin,coordinates_km[:,:2]),('d',self.destination,coordinates_km[:,2:4])]:
            for level in levels:results.append(dict(side=side,step_m=level.step,**level.fit_confidence(coords*1000)))
        return results
class SpatialFieldJoint(JointRouteTimeMixture):
    def __init__(self,mean,std,keep,bins,anchors,anchor_indices,geometry_indices,clock_indices,k=16,width=512,mode='spatial'):
        super().__init__(mean,std,keep,bins,anchors,anchor_indices,geometry_indices,k,width)
        self.field_mode=mode
        if mode=='spatial':self.field=MultiscaleODField(clock_indices)
        elif mode=='capacity':
            self.field=nn.Sequential(nn.Linear(len(keep),1220),nn.SiLU(),nn.Linear(1220,1220),nn.SiLU(),nn.Linear(1220,248))
            nn.init.zeros_(self.field[-1].weight);nn.init.zeros_(self.field[-1].bias)
        else:raise ValueError(mode)
    def forward(self,x):
        delta=self.field(x) if self.field_mode=='spatial' else self.field((x[:,self.keep]-self.mean)/self.std)
        hook=self.backbone.output.register_forward_hook(lambda module,args,value:value+delta[:,None,:])
        try:return super().forward(x)
        finally:hook.remove()
