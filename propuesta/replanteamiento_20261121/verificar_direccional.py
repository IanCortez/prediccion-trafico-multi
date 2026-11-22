from base_replanteo import *
from nucleo_direccional import directional_arrival
import torch
torch.set_num_threads(2);D=torch.float64;H=6;nb=torch.tensor([[1,2],[0,0],[0,0]]);policy=torch.tensor([[[.25,.75],[1.,0],[1.,0]]],dtype=D);wm=torch.zeros(1,3,H-1,dtype=D);wm[:,:,0]=1;ew=torch.zeros(1,2,H,dtype=D);ew[0,0,0]=1;ew[0,1,1]=1;ep=torch.ones(1,2,dtype=D);o=torch.tensor([0]);ends=torch.tensor([[1,2]])
with torch.no_grad():p=directional_arrival(policy,wm,ew,ep,nb,o,ends)
ref=torch.tensor([[0,.25,.75,0,0,0]],dtype=D);assert (p-ref).abs().max()<1e-10
logit=torch.tensor([0.],dtype=D,requires_grad=True);pp=torch.cat([logit.sigmoid(),1-logit.sigmoid()]);pol=torch.cat([pp[None,None,:],policy[:,1:,:]],1);p=directional_arrival(pol,wm,ew,ep,nb,o,ends);p[0,1].backward();assert abs(float(logit.grad)-.25)<1e-10
with torch.no_grad():fast=directional_arrival(pol,wm,ew,ep,nb,o,ends)
assert (fast-p.detach()).abs().max()<1e-10
# Destination revisits: absorption0.5 at t1,0.25 at t3,0.125 at t5.
ew[0,1]=0;ew[0,1,0]=1;ep*=.5;policy[0,0]=torch.tensor([1.,0],dtype=D)
with torch.no_grad():p=directional_arrival(policy,wm,ew,ep,nb,o,ends)
assert (p-torch.tensor([[0,.5,0,.25,0,.125]],dtype=D)).abs().max()<1e-10
# Query-dependent transition tables and batch permutation.
nb2=nb[None].repeat(2,1,1);nb2[1,0]=torch.tensor([2,1]);p2=directional_arrival(policy.repeat(2,1,1),wm.repeat(2,1,1),ew.repeat(2,1,1),ep.repeat(2,1),nb2,o.repeat(2),ends.repeat(2,1));rev=torch.tensor([1,0]);p3=directional_arrival(policy.repeat(2,1,1)[rev],wm.repeat(2,1,1)[rev],ew.repeat(2,1,1)[rev],ep.repeat(2,1)[rev],nb2[rev],o.repeat(2)[rev],ends.repeat(2,1)[rev]);assert (p2[rev]-p3).abs().max()<1e-10
write(OUT/'verificacion_direccional_cpu.json',dict(passed=True,multiple_directional_end_states=True,analytical_path_mixture_and_time_convolution=True,return_loops_mass_conserved=True,analytical_gradient=True,fast_and_grad_match=True,batch_permutation=True,query_actual_direction_used=False,Test_access=False));print('Directional absorption analytical checks passed.')

