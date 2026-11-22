"""15-second autoregressive path generation; only generated prefixes at inference."""
import torch
from torch import nn
from modelo_dir_fases import PhaseDirectionalArrival

def counter_uniform(seed,particle,step,stream=0):
    mask=0xffffffff
    v=(seed*0x9e3779b1+particle*0x85ebca77+int(step)*0xc2b2ae3d+int(stream)*0x27d4eb2f)&mask
    v=((v^(v>>16))*0x85ebca6b)&mask
    v=((v^(v>>13))*0xc2b2ae35)&mask
    v=v^(v>>16)
    return (v.to(torch.float64)+.5)/4294967296.

class NativePrefixArrival(PhaseDirectionalArrival):
    def __init__(self,stats,grid,graph,distance,power,counts,initialization):
        super().__init__(stats,grid,graph,distance,power,counts,initialization)
        self.prefix_rnn=nn.GRU(66,64,batch_first=True)
        self.history_projection=nn.Linear(64,64,bias=False)
        nn.init.zeros_(self.history_projection.weight)
        self.stay_head=nn.Linear(64,1)
        self.stop_head=nn.Linear(64,1)
        nn.init.zeros_(self.stay_head.weight)
        nn.init.zeros_(self.stay_head.bias)
        nn.init.zeros_(self.stop_head.weight)
        nn.init.constant_(self.stop_head.bias,-3.)
        for module in [self.od_head,self.wait_head,self.end_head,self.end_wait_head,self.end_direction]:
            for p in module.parameters():p.requires_grad_(False)
        self.mix_logit.requires_grad_(False)

    def load_phase_initialization(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        assert not unexpected
        assert set(missing)=={'prefix_rnn.weight_ih_l0','prefix_rnn.weight_hh_l0','prefix_rnn.bias_ih_l0','prefix_rnn.bias_hh_l0','history_projection.weight','stay_head.weight','stay_head.bias','stop_head.weight','stop_head.bias'}
        return missing

    def state_representation(self,q,geo,cell,destination,heading):
        current=self.centers[cell]
        dest=self.centers[destination]
        road=self.distance[cell,destination]
        road=torch.where(torch.isfinite(road),road,(current-dest).square().sum(-1).sqrt()*1.5)
        rel=torch.cat([current/5000,(current-geo[:,:2]*1000)/5000,(current-geo[:,2:4]*1000)/5000,road[:,None]/5000],-1)
        return self.state_encoder(torch.cat([q,self.cell(cell),rel,self.heading(heading)],-1))

    def action_probabilities(self,h,geo,cell,destination,heading,excluded_fold=None):
        neighbors=self.neighbors[cell]
        current=self.centers[cell]
        dest=self.centers[destination]
        d=self.distance[cell,destination]
        d=torch.where(torch.isfinite(d),d,(current-dest).square().sum(-1).sqrt()*1.5)
        next_center=self.centers[neighbors.clamp_min(0)]
        nd=self.distance[neighbors.clamp_min(0),destination[:,None]]
        nd=torch.where(torch.isfinite(nd),nd,(next_center-dest[:,None]).square().sum(-1).sqrt()*1.5)
        progress=((nd-d[:,None])/1000).clamp(-2,2)
        mask=self.allowed[cell]
        logits=self.route_head(h).float()-nn.functional.softplus(self.progress_log_scale)*progress
        prior=self.prior_probabilities(geo,cell,heading,excluded_fold)
        logits=logits+nn.functional.softplus(self.prior_log_strength).clamp_max(3)*prior.clamp_min(1e-30).log()
        route=logits.masked_fill(~mask,-1e9).softmax(-1)*mask
        stay=self.stay_head(h).float().sigmoid()
        motion=torch.cat([stay,(1-stay)*route],-1)
        motion=motion/motion.sum(-1,keepdim=True).clamp_min(1e-12)
        stop=self.stop_head(h).float().sigmoid()[:,0]*(cell==destination)
        return motion,stop

    def teacher_probabilities(self,x,origin,destination,cell,heading,age,excluded_fold=None):
        B,L=cell.shape
        q,*_=self.encode_od(x,origin,destination)
        def repeat(value):
            return value[:,None].expand(B,L,*value.shape[1:]).reshape(B*L,*value.shape[1:])
        flatcell=cell.reshape(-1)
        flathead=heading.reshape(-1)
        local=self.state_representation(repeat(q),repeat(x[:,:4]),flatcell,repeat(destination),flathead).reshape(B,L,64)
        elapsed=torch.arange(L,device=x.device,dtype=local.dtype)[None].expand(B,-1)/120
        inputs=torch.cat([local,age[:,:,None].to(local.dtype)/120,elapsed[:,:,None]],-1)
        history,_=self.prefix_rnn(inputs,q[None])
        state=(local+self.history_projection(history)).reshape(B*L,64)
        folds=None if excluded_fold is None else repeat(excluded_fold)
        motion,stop=self.action_probabilities(state,repeat(x[:,:4]),flatcell,repeat(destination),flathead,folds)
        return motion.reshape(B,L,49),stop.reshape(B,L)

    def rollout_step(self,q,geo,cell,destination,heading,age,elapsed,hidden):
        local=self.state_representation(q,geo,cell,destination,heading)
        extra=torch.stack([age.to(local.dtype)/120,torch.full_like(age,elapsed,dtype=local.dtype)/120],-1)
        hist,hidden=self.prefix_rnn(torch.cat([local,extra],-1)[:,None],hidden)
        state=local+self.history_projection(hist[:,0])
        motion,stop=self.action_probabilities(state,geo,cell,destination,heading)
        return motion,stop,hidden

    @torch.no_grad()
    def generated_arrival(self,x,origin,destination,query_seed,particles=128,horizon=120,stream=0):
        assert not self.training
        B=len(x);P=particles
        q,*_=self.encode_od(x,origin,destination)
        def expand(value):
            return value[:,None].expand(B,P,*value.shape[1:]).reshape(B*P,*value.shape[1:])
        qq=expand(q);geo=expand(x[:,:4]);dest=expand(destination)
        cell=expand(origin);heading=torch.full_like(cell,4);age=torch.zeros_like(cell)
        seed=expand(query_seed);particle=torch.arange(P,device=x.device).repeat(B)
        hidden=qq[None].contiguous()
        survival=torch.ones(B*P,device=x.device)
        raw=torch.zeros(B,horizon,device=x.device)
        for t in range(horizon):
            motion,stop,hidden=self.rollout_step(qq,geo,cell,dest,heading,age,t,hidden)
            raw[:,t]=(survival*stop).reshape(B,P).mean(-1)
            survival=survival*(1-stop)
            if t==horizon-1:break
            u=counter_uniform(seed,particle,t,stream)
            cdf=motion.to(torch.float64).cumsum(-1)
            cdf=cdf/cdf[:,-1:]
            action=(cdf<u[:,None]).sum(-1).clamp_max(48)
            move=action>0
            route_action=(action-1).clamp_min(0)
            nxt=self.neighbors[cell,route_action]
            # Zero-probability invalid edges cannot be sampled; catch corruption.
            cell=torch.where(move,nxt,cell)
            heading=torch.where(move,self.outgoing_heading[route_action],heading)
            age=torch.where(move,torch.zeros_like(age),age+1)
        return raw,survival.reshape(B,P).mean(-1)
