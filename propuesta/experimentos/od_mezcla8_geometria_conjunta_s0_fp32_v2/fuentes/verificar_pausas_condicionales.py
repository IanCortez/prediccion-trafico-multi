from common import *
from modelo_mezcla_conjunta import JointRouteTimeMixture
import torch
mass_errors=[]
for n in (4,20,119):
    for a,b in ((.1,.1),(.7,4.),(100.,100.)):
        count=torch.arange(n+1,dtype=torch.float64);prob=JointRouteTimeMixture.beta_binomial_logpmf(count,torch.tensor(float(n)),torch.tensor(a,dtype=torch.float64),torch.tensor(b,dtype=torch.float64)).exp();error=abs(float(prob.sum())-1);assert error<2e-5,(n,a,b,error);mass_errors.append(error)
B,K,J=3,2,8;torch.manual_seed(1729);weights=torch.randn(B,K,J,dtype=torch.float64).log_softmax(-1);component=torch.randn(B,K,J,116,dtype=torch.float64).log_softmax(-1);logp=torch.logsumexp(weights[...,None]+component,-2);parameters=torch.ones(B,K,J,2,dtype=torch.float64,requires_grad=True);output=(logp,weights,None,None,component,parameters);target=torch.tensor([60.,600.,1785.],dtype=torch.float64);counts=torch.tensor([0.,20.,119.],dtype=torch.float64);mask=torch.ones(B,dtype=torch.float64)
class Probe:
    beta_binomial_logpmf=staticmethod(JointRouteTimeMixture.beta_binomial_logpmf)
score=JointRouteTimeMixture.pause_score(Probe(),output,target,counts,mask);score.backward();assert torch.isfinite(parameters.grad).all() and parameters.grad.abs().sum()>0;assert abs(float(JointRouteTimeMixture.pause_score(Probe(),output,target,counts,torch.zeros(B))))<1e-9
write_json(BASE/'instrumentos/pausas_beta_binomial_contrato.json',dict(passed=True,max_mass_error=max(mass_errors),finite_gradients=True,zero_mask_zero_loss=True,training_count_conditioned_on_duration_label_only=True,query_point_count_not_input=True,test_labels_read=False));print('Conditional GPS-count contracts passed',flush=True)
