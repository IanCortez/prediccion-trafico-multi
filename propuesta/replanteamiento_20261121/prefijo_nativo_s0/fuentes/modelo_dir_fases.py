"""Directional route prior with a compact shifted-geometric residence process."""
import math
import numpy as np
import torch
from torch import nn
from modelo_dir_prior import PriorDirectionalArrival
from nucleo_fases import phase_arrival,geometric_wait_pmf

class PhaseDirectionalArrival(PriorDirectionalArrival):
    def __init__(self,stats,grid,graph,distance,power,counts,initialization):
        super().__init__(stats,grid,graph,distance,power,counts)
        self.phase_lags=tuple(initialization['lags'])
        self.K=len(self.phase_lags)
        self.wait_head=nn.Linear(64,2*self.K)
        nn.init.zeros_(self.wait_head.weight)
        weights=torch.tensor(initialization['weights'],dtype=torch.float32)
        hazards=torch.tensor(initialization['hazards'],dtype=torch.float32)
        with torch.no_grad():
            self.wait_head.bias.copy_(torch.cat([weights.log(),torch.logit(hazards)]))

    def phase_parameters(self,q,geo,cell_ids,destination,heading,excluded_fold=None):
        current=self.centers[cell_ids]
        dest=self.centers[destination]
        road_distance=self.distance[cell_ids,destination]
        fallback=(current-dest).square().sum(-1).sqrt()*1.5
        road_distance=torch.where(torch.isfinite(road_distance),road_distance,fallback)
        relative=torch.cat([
            current/5000,
            (current-geo[:,:2]*1000)/5000,
            (current-geo[:,2:4]*1000)/5000,
            road_distance[:,None]/5000,
        ],-1)
        h=self.state_encoder(torch.cat([q,self.cell(cell_ids),relative,self.heading(heading)],-1))
        neighbors=self.neighbors[cell_ids]
        next_center=self.centers[neighbors.clamp_min(0)]
        next_distance=self.distance[neighbors.clamp_min(0),destination[:,None]]
        fallback=(next_center-dest[:,None]).square().sum(-1).sqrt()*1.5
        next_distance=torch.where(torch.isfinite(next_distance),next_distance,fallback)
        progress=((next_distance-road_distance[:,None])/1000).clamp(-2,2)
        logits=self.route_head(h).float()-nn.functional.softplus(self.progress_log_scale)*progress
        mask=self.allowed[cell_ids]
        route=logits.masked_fill(~mask,-1e9).softmax(-1)*mask
        prior=self.prior_probabilities(geo,cell_ids,heading,excluded_fold)
        strength=nn.functional.softplus(self.prior_log_strength).clamp_max(3)
        logits=route.clamp_min(1e-30).log()+strength*prior.clamp_min(1e-30).log()
        route=logits.masked_fill(~mask,-1e9).softmax(-1)*mask
        residence=self.wait_head(h).float()
        weights=residence[:,:self.K].softmax(-1)
        hazards=residence[:,self.K:].sigmoid().clamp(.001,.999)
        return route,weights,hazards

    def state_parameters(self,q,geo,cell_ids,destination,heading,excluded_fold=None):
        route,weights,hazards=self.phase_parameters(q,geo,cell_ids,destination,heading,excluded_fold)
        return route,geometric_wait_pmf(weights,hazards,self.phase_lags,self.H)

    def process_parameters(self,x,origin=None,destination=None,excluded_fold=None):
        q,o,d,p0,e,we=self.encode_od(x,origin,destination)
        B=len(x)
        S=self.N*4+1
        repeated_cells=torch.arange(self.N,device=x.device).repeat_interleave(4)
        cells=torch.cat([repeated_cells[None].expand(B,-1),o[:,None]],1).reshape(-1)
        repeated_heading=torch.arange(4,device=x.device).repeat(self.N)
        heading=torch.cat([repeated_heading[None].expand(B,-1),torch.full((B,1),4,device=x.device)],1).reshape(-1)
        query=q[:,None].expand(B,S,64).reshape(-1,64)
        geo=x[:,:4][:,None].expand(B,S,4).reshape(-1,4)
        dest=d[:,None].expand(B,S).reshape(-1)
        folds=None if excluded_fold is None else excluded_fold[:,None].expand(B,S).reshape(-1)
        policies,weights,hazards=[],[],[]
        for first in range(0,len(cells),32768):
            last=first+32768
            pf,wf,hf=self.phase_parameters(query[first:last],geo[first:last],cells[first:last],dest[first:last],heading[first:last],None if folds is None else folds[first:last])
            policies.append(pf)
            weights.append(wf)
            hazards.append(hf)
        terminal_heading=torch.arange(5,device=x.device)[None].expand(B,-1).reshape(-1)
        end_probability,end_wait=self.end_parameters(e[:,None].expand(B,5).reshape(-1),we[:,None].expand(B,5,120).reshape(-1,120),terminal_heading)
        end_probability=end_probability.reshape(B,5)*torch.cat([torch.ones(B,4,device=x.device),(o==d).float()[:,None]],1)
        end_wait=end_wait.reshape(B,5,120)
        initial_neighbors=torch.where(self.neighbors[o]>=0,self.neighbors[o]*4+self.outgoing_heading[None],-1)
        neighbors=torch.cat([self.directed_neighbors[None].expand(B,-1,-1),initial_neighbors[:,None]],1)
        initial=torch.full_like(o,S-1)
        end_states=torch.cat([d[:,None]*4+torch.arange(4,device=x.device)[None],torch.full((B,1),S-1,device=x.device,dtype=torch.long)],1)
        return torch.cat(policies).reshape(B,S,48),torch.cat(weights).reshape(B,S,self.K),torch.cat(hazards).reshape(B,S,self.K),end_wait,end_probability,neighbors,initial,end_states,p0

    def posterior(self,x,origin=None,destination=None,excluded_fold=None,reference_probability=None):
        policy,weights,hazards,we,e,neighbors,initial,ends,p0=self.process_parameters(x,origin,destination,excluded_fold)
        raw=phase_arrival(policy,weights,hazards,self.phase_lags,we,e,neighbors,initial,ends)
        mass=raw[:,4:120].sum(-1)
        path=raw[:,4:120]/mass[:,None].clamp_min(1e-12)
        if reference_probability is not None:
            assert reference_probability.shape==p0.shape
            p0=reference_probability
        alpha=self.mix_logit.sigmoid()*mass/(mass+.01)
        p=(1-alpha[:,None])*p0+alpha[:,None]*path
        p=p/p.sum(-1,keepdim=True)
        return p,mass,p0,path

    def load_shared_state(self,state):
        current=self.state_dict()
        shared={k:v for k,v in state.items() if k in current and v.shape==current[k].shape}
        missing,unexpected=self.load_state_dict(shared,strict=False)
        assert set(missing)=={'wait_head.weight','wait_head.bias'} and not unexpected
        return {'loaded_parameters_and_buffers':len(shared),'new_head':missing}

