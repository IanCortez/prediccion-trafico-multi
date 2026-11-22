"""Explicit total time = movement + low-displacement intervals, with a latent mixture."""
import torch,math
from torch import nn
import tabm,rtdl_num_embeddings as rtdl

class DecomposedTime(nn.Module):
 def __init__(self,mean,std,keep,bins,anchors,anchor_indices,k=16,width=512):
  super().__init__();self.components=4
  for name,value,dtype in [('mean',mean,torch.float32),('std',std,torch.float32),('keep',keep,torch.long),('anchors',anchors,torch.float32),('anchor_indices',anchor_indices,torch.long)]:self.register_buffer(name,torch.as_tensor(value,dtype=dtype))
  self.register_buffer('anchor_scale',torch.tensor([600,600,600,1000.],dtype=torch.float32));self.register_buffer('duration_grid',torch.arange(60,1786,15,dtype=torch.float32));self.register_buffer('movement_log_edges',(torch.arange(.5,120,1)*15).log());counts=torch.arange(120,dtype=torch.float32);prior=(1-math.exp(-1/8))*torch.exp(-counts/8);prior=prior/prior.sum()*.5;prior[0]+=.5;self.register_buffer('log_stop_prior',prior.log())
  emb=rtdl.PiecewiseLinearEmbeddings(bins,d_embedding=16,activation=False,version='B');self.backbone=tabm.TabM.make(n_num_features=len(keep),num_embeddings=emb,d_out=4*123,k=k,d_block=width,n_blocks=2,dropout=.1,arch_type='tabm')
 def forward(self,x):
  z=(x[:,self.keep]-self.mean)/self.std;raw=self.backbone(z).reshape(len(x),-1,4,123).float();physical=x[:,self.anchor_indices].float()*self.anchor_scale;baseline=self.anchors[:,0]+self.anchors[:,1]*physical.clamp_min(.1).log();loc=baseline[:,None,:]+math.log(.8)+2*torch.tanh(raw[...,0]/2);sigma=(.12+nn.functional.softplus(raw[...,1]-1.5)).clamp_max(1.5);cdf=torch.special.ndtr((self.movement_log_edges-loc[...,None])/sigma[...,None]);pm=torch.cat([cdf[...,:1],cdf[...,1:]-cdf[...,:-1]],-1).clamp_min(1e-7);pm=pm/pm.sum(-1,keepdim=True);logs=(raw[...,3:]+self.log_stop_prior).log_softmax(-1);ps=logs.exp();weights=raw[...,2].log_softmax(-1)
  # Linear convolution of the two 120-bin PMFs. 256 avoids circular aliasing.
  conv=torch.fft.irfft(torch.fft.rfft(pm,n=256)*torch.fft.rfft(ps,n=256),n=256)[...,:239].clamp_min(1e-10);marginal=(conv[...,4:120]*weights.exp()[...,None]).sum(-2);Z=marginal.sum(-1);logp=marginal.clamp_min(1e-10).log()-Z.clamp_min(1e-10).log()[...,None]
  return logp,pm.log(),logs,weights,Z.clamp_min(1e-10).log()
 def conditional_count_score(self,output,total_n,low_n,reliable):
  logp,logm,logs,logw,logZ=output;B,K,J=logw.shape;m=(total_n-low_n).long();s=low_n.long();midx=m[:,None,None,None].expand(B,K,J,1);sidx=s[:,None,None,None].expand(B,K,J,1);joint=torch.logsumexp(logw+logm.gather(-1,midx)[...,0]+logs.gather(-1,sidx)[...,0],-1)-logZ;marginal=logp.gather(-1,(total_n.long()-4)[:,None,None].expand(B,K,1))[...,0];score=-(joint-marginal).mean(1)
  return (score*reliable).sum()/reliable.sum().clamp_min(1)
 def median(self,logp):
  cdf=logp.exp().mean(1).cumsum(-1);return self.duration_grid[(cdf<.5).sum(-1).clamp_max(115)]

