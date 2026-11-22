import torch
from torch import nn
import tabm,rtdl_num_embeddings as rtdl
class RouteMixtureTabM(nn.Module):
    """Conditional duration distribution over four cost hypotheses.
    The evaluated query's observed trajectory is never accepted by forward().
    """
    def __init__(self,mean,std,keep,bins,anchors,anchor_indices,k=16,width=512):
        super().__init__()
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32));self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32));self.register_buffer('keep',torch.as_tensor(keep,dtype=torch.long))
        self.register_buffer('anchors',torch.as_tensor(anchors,dtype=torch.float32));self.register_buffer('anchor_indices',torch.as_tensor(anchor_indices,dtype=torch.long))
        self.register_buffer('duration_grid',torch.arange(60,1786,15,dtype=torch.float32));self.register_buffer('log_edges',torch.arange(67.5,1785,15,dtype=torch.float32).log())
        embeddings=rtdl.PiecewiseLinearEmbeddings(bins,d_embedding=16,activation=False,version='B')
        self.backbone=tabm.TabM.make(n_num_features=len(keep),num_embeddings=embeddings,d_out=12,k=k,d_block=width,n_blocks=2,dropout=.1,arch_type='tabm')
    def forward(self,x):
        z=(x[:,self.keep]-self.mean)/self.std
        raw=self.backbone(z).reshape(len(x),-1,4,3).float()
        physical=x[:,self.anchor_indices].float()*torch.tensor([600,600,600,1000.],device=x.device)
        base=self.anchors[:,0]+self.anchors[:,1]*physical.clamp_min(.1).log()
        loc=base[:,None,:]+1.5*torch.tanh(raw[...,0]/1.5)
        sigma=(.15+torch.nn.functional.softplus(raw[...,1]-1.5)).clamp_max(1.5)
        weights=raw[...,2].log_softmax(-1)
        cdf=torch.special.ndtr((self.log_edges-loc[...,None])/sigma[...,None])
        prob=torch.cat([cdf[...,:1],cdf[...,1:]-cdf[...,:-1],1-cdf[...,-1:]],-1).clamp_min(1e-7)
        prob=prob/prob.sum(-1,keepdim=True)
        logprob=torch.logsumexp(prob.log()+weights[...,None],dim=-2)
        return logprob,weights
    def quantiles(self,logprob,levels=(.1,.5,.9),mode='discrete'):
        distribution=logprob.exp().mean(1);cdf=distribution.cumsum(-1)
        if mode=='discrete':return torch.stack([self.duration_grid[(cdf<q).sum(-1).clamp_max(115)] for q in levels],-1)
        if mode!='interpolated':raise ValueError(mode)
        result=[]
        for q in levels:
            index=(cdf<q).sum(-1).clamp_max(115);before=torch.where(index>0,cdf.gather(-1,(index-1).clamp_min(0)[:,None])[:,0],torch.zeros_like(cdf[:,0]));mass=distribution.gather(-1,index[:,None])[:,0]
            lower=(self.duration_grid[index]-7.5).clamp_min(60);upper=(self.duration_grid[index]+7.5).clamp_max(1785);alpha=((q-before)/mass.clamp_min(1e-7)).clamp(0,1);result.append(lower+(upper-lower)*alpha)
        return torch.stack(result,-1)

def make_duration_model(checkpoint,config):
    if config.get('architecture')=='tabm_joint_route_time':
        from modelo_mezcla_conjunta import JointRouteTimeMixture
        return JointRouteTimeMixture(checkpoint['mean'],checkpoint['std'],checkpoint['keep'],checkpoint['bins'],checkpoint['anchors'],checkpoint['anchor_indices'],checkpoint['geometry_indices'],config['k'],config['width'])
    return RouteMixtureTabM(checkpoint['mean'],checkpoint['std'],checkpoint['keep'],checkpoint['bins'],checkpoint['anchors'],checkpoint['anchor_indices'],config['k'],config['width'])
