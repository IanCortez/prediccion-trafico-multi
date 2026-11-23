"""Global joint time/geometry denoising: no autoregressive road prefix at query time."""
import math
import torch
from torch import nn
from torch.nn import functional as F

def geometry_transform(phi):
    assert phi.shape[-1]==90
    return torch.cat([torch.log1p(phi[...,:9].clamp_min(0)),phi[...,9:12],
                      torch.sqrt(phi[...,12:].clamp_min(0))],-1)

class ConditionalBlock(nn.Module):
    def __init__(self,width):
        super().__init__();self.norm=nn.LayerNorm(width)
        self.modulation=nn.Linear(width,2*width)
        self.net=nn.Sequential(nn.Linear(width,2*width),nn.SiLU(),nn.Linear(2*width,width))
    def forward(self,h,condition):
        scale,shift=self.modulation(condition).chunk(2,-1)
        return h+self.net(self.norm(h)*(1+.1*scale.tanh())+shift)

class GlobalJointDiffusion(nn.Module):
    def __init__(self,xmean,xstd,target_mean,target_std,seed=20261122,width=384):
        super().__init__();self.timesteps=1000;self.width=width
        for key,value in [('xmean',xmean),('xstd',xstd),('target_mean',target_mean),('target_std',target_std)]:
            self.register_buffer(key,torch.as_tensor(value,dtype=torch.float32))
        assert self.target_mean.shape==(91,) and len(self.xmean)==216
        self.condition=nn.Sequential(nn.Linear(216,width),nn.SiLU(),nn.LayerNorm(width),
                                    nn.Linear(width,width),nn.SiLU())
        self.time_net=nn.Sequential(nn.Linear(64,width),nn.SiLU(),nn.Linear(width,width))
        self.input=nn.Linear(91,width);self.blocks=nn.ModuleList([ConditionalBlock(width) for _ in range(4)])
        self.output=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,91))
        nn.init.normal_(self.output[-1].weight,std=.005);nn.init.zeros_(self.output[-1].bias)
        tau=torch.linspace(0,1,self.timesteps+1,dtype=torch.float64)
        alpha=(torch.cos((tau+.008)/1.008*math.pi/2)**2)/(math.cos(.008/1.008*math.pi/2)**2)
        self.register_buffer('alpha_bar',alpha.clamp(1e-5,1).float())
        self.register_buffer('time_frequency',torch.logspace(0,3,32))
        engine=torch.quasirandom.SobolEngine(91,scramble=True,seed=seed)
        u=engine.draw(256).clamp(1e-6,1-1e-6)
        self.register_buffer('base_noise',math.sqrt(2)*torch.erfinv(2*u-1))
        self.inference_levels=tuple(round(self.timesteps*(1-k/32)) for k in range(33))

    def public_context(self,features):
        assert features.ndim==2 and features.shape[1]==216
        return self.condition(((features-self.xmean)/self.xstd).clamp(-30,30))

    def velocity(self,noisy,noise_level,context):
        phase=(noise_level.float()/self.timesteps)[:,None]*self.time_frequency[None]*2*math.pi
        condition=context+self.time_net(torch.cat([phase.sin(),phase.cos()],-1))
        h=self.input(noisy)
        for block in self.blocks:h=block(h,condition)
        return self.output(h)

    def private_loss(self,features,original_T_seconds,phi,available,teacher_confidence,noise_level=None,noise=None):
        """PRIVATE labels; primary time loss includes EVERY query, even without geometry."""
        B=len(features);assert original_T_seconds.shape==(B,) and phi.shape==(B,90)
        target=torch.cat([original_T_seconds.log()[:,None],geometry_transform(phi)],1)
        target=(target-self.target_mean)/self.target_std
        target=torch.cat([target[:,:1],torch.where(available[:,None],target[:,1:],torch.zeros_like(target[:,1:]))],1)
        if noise_level is None:noise_level=torch.randint(1,self.timesteps+1,(B,),device=features.device)
        if noise is None:noise=torch.randn_like(target)
        a=self.alpha_bar[noise_level].sqrt()[:,None];s=(1-self.alpha_bar[noise_level]).sqrt()[:,None]
        noisy=a*target+s*noise;desired=a*noise-s*target
        predicted=self.velocity(noisy,noise_level,self.public_context(features))
        time=(predicted[:,0]-desired[:,0]).square().mean()
        weight=available.float()*teacher_confidence.clamp(0,1)
        geometry=((predicted[:,1:]-desired[:,1:]).square().mean(1)*weight).sum()/weight.sum().clamp_min(1)
        return time+.2*geometry,dict(time=time.detach(),geometry=geometry.detach(),
                                     primary_examples=B,geometry_examples=int(available.sum()))

    @torch.no_grad()
    def sample_latents(self,features,samples=64):
        assert not self.training and samples in (64,256)
        B=len(features);context=self.public_context(features).repeat_interleave(samples,0)
        z=self.base_noise[:samples][None].expand(B,-1,-1).reshape(B*samples,91).clone()
        for current,next_level in zip(self.inference_levels[:-1],self.inference_levels[1:]):
            levels=torch.full((len(z),),current,device=z.device,dtype=torch.long)
            a=self.alpha_bar[current].sqrt();s=(1-self.alpha_bar[current]).sqrt()
            v=self.velocity(z,levels,context)
            clean=a*z-s*v;epsilon=s*z+a*v
            z=self.alpha_bar[next_level].sqrt()*clean+(1-self.alpha_bar[next_level]).sqrt()*epsilon
        return z.reshape(B,samples,91)

    @torch.no_grad()
    def sample_times(self,features,samples=64):
        # Latent T starts as independent Gaussian noise, never as a true query label.
        latent=self.sample_latents(features,samples)
        logT=latent[...,0]*self.target_std[0]+self.target_mean[0]
        return logT.clamp(math.log(60),math.log(1785)).exp().clamp(60,1785)

    @torch.no_grad()
    def quantiles(self,features,samples=64):
        times=self.sample_times(features,samples)
        return torch.quantile(times,torch.tensor([.1,.5,.9],device=times.device),dim=1).T
