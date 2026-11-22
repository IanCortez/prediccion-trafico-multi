import torch,math
from torch import nn
class ResidualBlock(nn.Module):
    def __init__(self,width,drop):
        super().__init__();self.net=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*2),nn.GELU(),nn.Dropout(drop),nn.Linear(width*2,width))
    def forward(self,x):return x+self.net(x)
class ODResidual(nn.Module):
    """Positive physical baseline plus shared spatial/temporal residual.
    No observed path or arrival information is accepted by this interface.
    """
    def __init__(self,mean,std,coef,width=256,drop=.05,grids=True,fourier=True):
        super().__init__()
        self.register_buffer("mean",torch.as_tensor(mean,dtype=torch.float32));self.register_buffer("std",torch.as_tensor(std,dtype=torch.float32))
        self.register_buffer("coef",torch.as_tensor(coef,dtype=torch.float32))
        self.scales=(.1,.25,.5,1.0) if grids else ()
        self.periods=(.1,.25,.5,1.,2.,4.,8.) if fourier else ()
        self.tables=nn.ModuleList([nn.Embedding(int(20/s)**2,8) for s in self.scales])
        for e in self.tables:nn.init.normal_(e.weight,std=.01)
        dim=2+4*len(self.periods)+8*len(self.scales)
        self.point=nn.Sequential(nn.Linear(dim,96),nn.GELU(),nn.Linear(96,64))
        self.joint=nn.Sequential(nn.Linear(4*64+len(mean),width),nn.GELU(),ResidualBlock(width,drop),ResidualBlock(width,drop),nn.LayerNorm(width),nn.Linear(width,1))
        nn.init.zeros_(self.joint[-1].weight);nn.init.zeros_(self.joint[-1].bias)
        # Explicit small local calibration; later shared source encoder can be frozen.
        self.local_bias=nn.Parameter(torch.zeros(()));self.local_gain=nn.Parameter(torch.ones(()))
    def encode_point(self,xy):
        parts=[xy/5]
        for s in self.periods:
            angle=xy*(2*math.pi/s);parts.extend([angle.sin(),angle.cos()])
        for s,table in zip(self.scales,self.tables):
            k=int(20/s);cell=torch.floor((xy+10)/s).long().clamp(0,k-1)
            parts.append(table(cell[:,0]*k+cell[:,1]))
        return self.point(torch.cat(parts,dim=1))
    def forward(self,x):
        o=self.encode_point(x[:,:2]);d=self.encode_point(x[:,2:4])
        z=torch.cat([o,d,o*d,o-d,(x-self.mean)/self.std],dim=1)
        correction=self.local_gain*self.joint(z).squeeze(1)+self.local_bias
        correction=3*torch.tanh(correction/3)
        baseline=self.coef[0]+self.coef[1]*torch.log((x[:,6]*1000).clamp_min(1))
        return torch.exp(baseline+correction)


class ODFiLMResidual(ODResidual):
    """Temporal conditioning of geometry, anchored to known map free-flow time."""
    def __init__(self,mean,std,coef,width=128,drop=.1,grids=True,fourier=True):
        super().__init__(mean,std,coef,width,drop,grids,fourier)
        del self.joint
        self.register_buffer("geo_columns",torch.tensor([i for i in range(len(mean)) if not 14<=i<24],dtype=torch.long))
        self.geometry=nn.Sequential(nn.Linear(256+len(mean)-10,width),nn.GELU(),ResidualBlock(width,drop))
        self.temporal=nn.Sequential(nn.Linear(10,64),nn.GELU(),nn.Linear(64,2*width))
        self.final=nn.Sequential(ResidualBlock(width,drop),nn.LayerNorm(width),nn.Linear(width,1))
        nn.init.zeros_(self.final[-1].weight);nn.init.zeros_(self.final[-1].bias)
    def forward(self,x):
        o=self.encode_point(x[:,:2]);d=self.encode_point(x[:,2:4]);normalized=(x-self.mean)/self.std
        h=self.geometry(torch.cat([o,d,o*d,o-d,normalized[:,self.geo_columns]],1))
        gamma,bias=self.temporal(normalized[:,14:24]).chunk(2,dim=1)
        h=h*(1+.2*torch.tanh(gamma))+bias
        r=self.local_gain*self.final(h).squeeze(1)+self.local_bias
        correction=3*torch.tanh(r/3)
        if len(self.coef)==3:
            baseline=self.coef[0]+self.coef[1]*torch.log((x[:,6]*1000).clamp_min(1))+self.coef[2]*torch.log((x[:,33]*600).clamp_min(1))
        else:
            baseline=self.coef[0]+self.coef[1]*torch.log((x[:,6]*1000).clamp_min(1))
        return torch.exp(baseline+correction)

def build_model(mean,std,coef,config):
    cls=ODFiLMResidual if config.get("architecture","residual")=="film" else ODResidual
    return cls(mean,std,coef,width=config["width"],drop=config["dropout"],grids=not config["no_grids"],fourier=not config["no_fourier"])

