"""Supervised latent semantic type; observed type is never a forward input."""
import torch
from torch import nn
class LatentTypeTilt(nn.Module):
    def __init__(self,mean,std):
        super().__init__()
        self.register_buffer('mean',mean.clone())
        self.register_buffer('std',std.clone())
        grid=torch.arange(60,1786,15,dtype=torch.float32)
        self.register_buffer('grid',grid)
        centers=torch.linspace(grid.log().min(),grid.log().max(),8)
        basis=torch.exp(-.5*((grid.log()[None,:]-centers[:,None])/.45)**2)
        self.register_buffer('basis',basis-basis.mean(1,keepdim=True))
        self.repr=nn.Sequential(nn.Linear(len(mean),64),nn.SiLU())
        self.out=nn.Linear(64,27)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,hidden,base_probability):
        # No observed class, duration, trajectory or trip length in this interface.
        raw=self.out(self.repr((hidden-self.mean)/self.std))
        gate=raw[:,:3].log_softmax(-1)
        coeff=1.5*torch.tanh(raw[:,3:].reshape(-1,3,8)/1.5)
        conditional=(base_probability.clamp_min(1e-9).log()[:,None,:]+coeff@self.basis).log_softmax(-1)
        marginal=torch.logsumexp(gate[:,:,None]+conditional,dim=1)
        return marginal,gate,conditional
def duration_median(logp):
    return 60+15*(logp.exp().cumsum(-1)<.5).sum(-1).clamp_max(115)
