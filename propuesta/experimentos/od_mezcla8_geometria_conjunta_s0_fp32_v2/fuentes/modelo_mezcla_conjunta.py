import torch,math
from torch import nn
import tabm,rtdl_num_embeddings as rtdl
from modelo_mezcla_rutas import RouteMixtureTabM
class JointRouteTimeMixture(nn.Module):
    """Eight latent route-duration components. Observed geometry is a training target only."""
    quantiles=RouteMixtureTabM.quantiles
    def __init__(self,mean,std,keep,bins,anchors,anchor_indices,geometry_indices,k=16,width=512):
        super().__init__();self.components=8;self.geo_dim=14
        for name,value,dtype in [('mean',mean,torch.float32),('std',std,torch.float32),('keep',keep,torch.long),('anchors',torch.as_tensor(anchors).repeat(2,1),torch.float32),('anchor_indices',torch.as_tensor(anchor_indices).repeat(2),torch.long),('geometry_indices',geometry_indices,torch.long)]:self.register_buffer(name,torch.as_tensor(value,dtype=dtype))
        assert self.geometry_indices.shape==(8,14);self.register_buffer('duration_grid',torch.arange(60,1786,15,dtype=torch.float32));self.register_buffer('log_edges',torch.arange(67.5,1785,15,dtype=torch.float32).log());self.register_buffer('anchor_scale',torch.tensor([600,600,600,1000.]*2))
        embeddings=rtdl.PiecewiseLinearEmbeddings(bins,d_embedding=16,activation=False,version='B');self.backbone=tabm.TabM.make(n_num_features=len(keep),num_embeddings=embeddings,d_out=8*(3+28),k=k,d_block=width,n_blocks=2,dropout=.1,arch_type='tabm')
    def forward(self,x):
        z=(x[:,self.keep]-self.mean)/self.std;raw=self.backbone(z).reshape(len(x),-1,8,31).float();physical=x[:,self.anchor_indices].float()*self.anchor_scale;base=self.anchors[:,0]+self.anchors[:,1]*physical.clamp_min(.1).log();loc=base[:,None,:]+1.5*torch.tanh(raw[...,0]/1.5);sigma=(.15+torch.nn.functional.softplus(raw[...,1]-1.5)).clamp_max(1.5);log_weights=raw[...,2].log_softmax(-1);cdf=torch.special.ndtr((self.log_edges-loc[...,None])/sigma[...,None]);prob=torch.cat([cdf[...,:1],cdf[...,1:]-cdf[...,:-1],1-cdf[...,-1:]],-1).clamp_min(1e-7);prob=prob/prob.sum(-1,keepdim=True);component_logp=prob.log();logp=torch.logsumexp(component_logp+log_weights[...,None],dim=-2)
        idx=self.geometry_indices;prior=x[:,idx.clamp_min(0)].float()*(idx>=0);geo_loc=prior[:,None]+3*torch.tanh(raw[...,3:17]/3);geo_sigma=(.03+torch.nn.functional.softplus(raw[...,17:31]-2)).clamp_max(4.)
        pause_parameters=(.1+torch.nn.functional.softplus(raw[...,3:5])).clamp_max(100.)
        return logp,log_weights,geo_loc,geo_sigma,component_logp,pause_parameters
    def geometry_score(self,output,target_seconds,geometry,mask):
        logp,weights,loc,sigma,component_logp=output[:5];label=((target_seconds-60)/15).long();B,K,J=weights.shape;time_logp=component_logp.gather(-1,label[:,None,None,None].expand(B,K,J,1))[...,0];marginal=logp.gather(-1,label[:,None,None].expand(B,K,1))[...,0];responsibility=weights+time_logp-marginal[...,None]
        log_g=(-.5*((geometry[:,None,None]-loc)/sigma).square()-sigma.log()-.5*math.log(2*math.pi)).sum(-1);score=-torch.logsumexp(responsibility+log_g,-1).mean(1)/14
        return (score*mask).sum()/mask.sum().clamp_min(1)

    @staticmethod
    def beta_binomial_logpmf(count,n,alpha,beta):
        count,n,alpha,beta=[value.to(torch.float64) for value in (count,n,alpha,beta)]
        return (torch.lgamma(n+1)-torch.lgamma(count+1)-torch.lgamma(n-count+1)+torch.lgamma(count+alpha)+torch.lgamma(n-count+beta)-torch.lgamma(n+alpha+beta)+torch.lgamma(alpha+beta)-torch.lgamma(alpha)-torch.lgamma(beta))
    def pause_score(self,output,target_seconds,low_displacement_count,mask):
        logp,weights,_,_,component_logp,parameters=output;label=((target_seconds-60)/15).long();B,K,J=weights.shape;time_logp=component_logp.gather(-1,label[:,None,None,None].expand(B,K,J,1))[...,0];marginal=logp.gather(-1,label[:,None,None].expand(B,K,1))[...,0];responsibility=weights+time_logp-marginal[...,None];n=(target_seconds[:,None,None]/15).round();count=low_displacement_count[:,None,None];log_counts=self.beta_binomial_logpmf(count,n,parameters[...,0],parameters[...,1]);score=-torch.logsumexp(responsibility+log_counts,-1).mean(1)
        return (score*mask).sum()/mask.sum().clamp_min(1)
