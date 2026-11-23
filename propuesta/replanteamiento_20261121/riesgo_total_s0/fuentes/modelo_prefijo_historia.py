"""Recurrent native generator with empirical temporal competing-risk priors."""
import torch
from torch import nn
from modelo_prefijo_nativo import NativePrefixArrival,counter_uniform

class HistoricalPrefixArrival(NativePrefixArrival):
    def __init__(self,stats,grid,graph,distance,power,counts,initialization,full_probability):
        super().__init__(stats,grid,graph,distance,power,counts,initialization)
        assert full_probability.shape==(self.N*120,120,3)
        self.register_buffer('historical_events',torch.as_tensor(full_probability,dtype=torch.float32),persistent=False)
        self.event_residual=nn.Linear(64,3)
        nn.init.zeros_(self.event_residual.weight);nn.init.zeros_(self.event_residual.bias)
        for module in [self.stay_head,self.stop_head]:
            for p in module.parameters():p.requires_grad_(False)

    def load_native_initialization(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        assert set(missing)=={'event_residual.weight','event_residual.bias'} and not unexpected
        return missing

    def route_probabilities(self,h,geo,cell,destination,heading,excluded_fold=None):
        neighbors=self.neighbors[cell]
        current=self.centers[cell];dest=self.centers[destination]
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
        return logits.masked_fill(~mask,-1e9).softmax(-1)*mask

    def action_with_history(self,h,geo,cell,destination,heading,event_prior,excluded_fold=None):
        route=self.route_probabilities(h,geo,cell,destination,heading,excluded_fold)
        logits=self.event_residual(h).float()+event_prior.clamp_min(1e-30).log()
        valid=torch.stack([torch.ones_like(cell,dtype=torch.bool),self.allowed[cell].any(-1),cell==destination],-1)
        event=logits.masked_fill(~valid,-1e9).softmax(-1)
        stop=event[:,2]
        motion=torch.cat([event[:,:1],event[:,1:2]*route],-1)
        motion=motion/motion.sum(-1,keepdim=True).clamp_min(1e-30)
        return motion,stop

    def teacher_probabilities(self,x,origin,destination,cell,heading,age,excluded_fold=None,event_prior=None):
        assert event_prior is not None,'Training must explicitly supply fold-excluded event history'
        B,L=cell.shape
        assert event_prior.shape==(B,L,3)
        q,*_=self.encode_od(x,origin,destination)
        def repeat(value):
            return value[:,None].expand(B,L,*value.shape[1:]).reshape(B*L,*value.shape[1:])
        local=self.state_representation(repeat(q),repeat(x[:,:4]),cell.reshape(-1),repeat(destination),heading.reshape(-1)).reshape(B,L,64)
        elapsed=torch.arange(L,device=x.device,dtype=local.dtype)[None].expand(B,-1)/120
        inputs=torch.cat([local,age[:,:,None].to(local.dtype)/120,elapsed[:,:,None]],-1)
        history,_=self.prefix_rnn(inputs,q[None])
        state=(local+self.history_projection(history)).reshape(B*L,64)
        folds=None if excluded_fold is None else repeat(excluded_fold)
        motion,stop=self.action_with_history(state,repeat(x[:,:4]),cell.reshape(-1),repeat(destination),heading.reshape(-1),event_prior.reshape(B*L,3),folds)
        return motion.reshape(B,L,49),stop.reshape(B,L)

    def historical_probability(self,cell,destination,heading,timestamp,elapsed,age):
        entry=timestamp+15*(int(elapsed)-age)
        weekday=(entry//86400+3)%7
        clock=(entry%86400)//14400+6*(weekday>=5)
        group=((cell*5+heading)*2+(cell==destination).long())*12+clock
        return self.historical_events[group,age.clamp(0,119)]

    def rollout_with_history(self,q,geo,cell,destination,heading,age,elapsed,hidden,timestamp):
        local=self.state_representation(q,geo,cell,destination,heading)
        extra=torch.stack([age.to(local.dtype)/120,torch.full_like(age,elapsed,dtype=local.dtype)/120],-1)
        hist,hidden=self.prefix_rnn(torch.cat([local,extra],-1)[:,None],hidden)
        state=local+self.history_projection(hist[:,0])
        prior=self.historical_probability(cell,destination,heading,timestamp,elapsed,age)
        motion,stop=self.action_with_history(state,geo,cell,destination,heading,prior)
        return motion,stop,hidden

    @torch.no_grad()
    def generated_arrival(self,x,origin,destination,query_seed,particles=128,horizon=120,stream=0,timestamp=None):
        assert timestamp is not None and timestamp.dtype==torch.int64 and not self.training
        B=len(x);P=particles
        q,*_=self.encode_od(x,origin,destination)
        def expand(value):
            return value[:,None].expand(B,P,*value.shape[1:]).reshape(B*P,*value.shape[1:])
        qq=expand(q);geo=expand(x[:,:4]);dest=expand(destination);ts=expand(timestamp)
        cell=expand(origin);heading=torch.full_like(cell,4);age=torch.zeros_like(cell)
        seed=expand(query_seed);particle=torch.arange(P,device=x.device).repeat(B)
        hidden=qq[None].contiguous()
        survival=torch.ones(B*P,device=x.device)
        raw=torch.zeros(B,horizon,device=x.device)
        for t in range(horizon):
            motion,stop,hidden=self.rollout_with_history(qq,geo,cell,dest,heading,age,t,hidden,ts)
            raw[:,t]=(survival*stop).reshape(B,P).mean(-1)
            survival=survival*(1-stop)
            if t==horizon-1:break
            u=counter_uniform(seed,particle,t,stream)
            cdf=motion.to(torch.float64).cumsum(-1)
            cdf=cdf/cdf[:,-1:]
            action=(cdf<u[:,None]).sum(-1).clamp_max(48)
            move=action>0;route_action=(action-1).clamp_min(0)
            nxt=self.neighbors[cell,route_action]
            cell=torch.where(move,nxt,cell)
            heading=torch.where(move,self.outgoing_heading[route_action],heading)
            age=torch.where(move,torch.zeros_like(age),age+1)
        return raw,survival.reshape(B,P).mean(-1)
