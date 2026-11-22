"""OD-conditioned trip-shared environment; no posterior effect from query labels."""
import math
import torch
from torch import nn
from modelo_dir_fases import PhaseDirectionalArrival
from nucleo_entorno_compartido import shared_environment_arrival,conditional_cohort

class SharedEnvironmentArrival(PhaseDirectionalArrival):
    def __init__(self,stats,grid,graph,distance,power,counts,initialization,environment_initialization):
        super().__init__(stats,grid,graph,distance,power,counts,initialization)
        self.environment_head=nn.Sequential(nn.Linear(len(stats['keep']),32),nn.SiLU(),nn.Linear(32,2))
        nn.init.zeros_(self.environment_head[-1].weight)
        location=float(environment_initialization['location'])
        sigma=max(float(environment_initialization['sigma']),.01)
        with torch.no_grad():
            self.environment_head[-1].bias.copy_(torch.tensor([math.atanh(max(-.99,min(.99,location/2))),math.log(math.expm1(sigma))]))
        nodes,prior=__import__('numpy').polynomial.hermite.hermgauss(4)
        self.register_buffer('environment_nodes',torch.tensor(nodes,dtype=torch.float32))
        self.register_buffer('environment_prior',torch.tensor(prior/math.sqrt(math.pi),dtype=torch.float32))
        # A new correction starts close to the protected model, with a distinct
        # optimizer learning-rate group for this scalar during training.
        with torch.no_grad():self.mix_logit.fill_(math.log(.02/.98))

    def environment_parameters(self,x):
        z=((x[:,self.keep]-self.mean)/self.std).clamp(-8,8)
        pars=self.environment_head(z).float()
        return 2*pars[:,0].tanh(),nn.functional.softplus(pars[:,1]).clamp_max(1.5)

    def posterior(self,x,origin=None,destination=None,excluded_fold=None,reference_probability=None):
        policy,weights,hazards,we,e,neighbors,initial,ends,p0=self.process_parameters(x,origin,destination,excluded_fold)
        location,sigma=self.environment_parameters(x)
        raw=shared_environment_arrival(policy,weights,hazards,self.phase_lags,we,e,neighbors,initial,ends,location,sigma,self.environment_nodes,self.environment_prior)
        path,mass=conditional_cohort(raw)
        if reference_probability is not None:
            assert reference_probability.shape==p0.shape
            p0=reference_probability
        alpha=self.mix_logit.sigmoid()*mass/(mass+.01)
        p=(1-alpha[:,None])*p0+alpha[:,None]*path
        return p/p.sum(-1,keepdim=True),mass,p0,path

    def load_independent_initialization(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        expected={'environment_nodes','environment_prior','environment_head.0.weight','environment_head.0.bias','environment_head.2.weight','environment_head.2.bias'}
        assert set(missing)==expected and not unexpected
        with torch.no_grad():self.mix_logit.fill_(math.log(.02/.98))
        return dict(missing_new_environment_parameters=missing)
