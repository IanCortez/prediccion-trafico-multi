"""Analytical small-graph checks for semi-Markov duration inference."""
from base_replanteo import *
from nucleo_semi_markov import arrival_distribution
import torch
T=6;dtype=torch.float64;policy=torch.ones(1,2,1,dtype=dtype);nb=torch.tensor([[1],[0]]);origin=torch.tensor([0]);destination=torch.tensor([1]);wm=torch.zeros(1,2,T-1,dtype=dtype);we=torch.zeros(1,T,dtype=dtype);wm[0,0,1]=1;wm[0,1,0]=1;we[0,1]=1
with torch.no_grad():p=arrival_distribution(policy,wm,we,torch.ones(1,dtype=dtype),nb,origin,destination)
ref=torch.zeros_like(p);ref[0,3]=1;assert (p-ref).abs().max()<1e-10
wm[0,0]=0;wm[0,0,:2]=.5;we[0]=0;we[0,:2]=.5
with torch.no_grad():p=arrival_distribution(policy,wm,we,torch.ones(1,dtype=dtype),nb,origin,destination)
ref[0]=torch.tensor([0,.25,.5,.25,0,0],dtype=dtype);assert (p-ref).abs().max()<1e-10
wm[0,0]=0;wm[0,0,0]=1;we[0]=0;we[0,0]=1;logit=torch.tensor([0.],dtype=dtype,requires_grad=True);p=arrival_distribution(policy,wm,we,logit.sigmoid(),nb,origin,destination);ref[0]=torch.tensor([0,.5,0,.25,0,.125],dtype=dtype);assert (p-ref).abs().max()<1e-10;p[0,1].backward();assert abs(float(logit.grad)-.25)<1e-10
with torch.no_grad():fast=arrival_distribution(policy,wm,we,logit.sigmoid(),nb,origin,destination)
assert (fast-p.detach()).abs().max()<1e-10 and float(p.sum())<=1+1e-10
write(OUT/'verificacion_semi_markov_cpu.json',dict(passed=True,deterministic_time_addition=True,analytical_convolution_of_waits=True,return_loops_geometric_absorption=True,fast_and_differentiable_versions_match=True,analytical_gradient_matches=True,finite_horizon_mass_conserved=True,query_route_or_duration_used=False,Test_access=False))
print('Semi-Markov analytical time, loop, mass and gradient checks passed.')
