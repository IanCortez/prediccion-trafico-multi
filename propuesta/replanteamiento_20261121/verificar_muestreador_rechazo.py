from base_replanteo import *
from muestreador_rechazo_nativo import sample_cohort_trajectories
from gradiente_crps_condicional import conditional_crps_weights,conditional_crps_surrogate
import torch
torch.set_num_threads(2);torch.manual_seed(67)
D=torch.float64
class TwoRouteModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits=torch.nn.Parameter(torch.tensor([-.4,.2,.8],dtype=D))
        nb=torch.full((4,48),-1,dtype=torch.long)
        nb[0,:3]=torch.tensor([2,1,3]);nb[1,0]=2
        self.register_buffer('neighbors',nb)
        self.register_buffer('outgoing_heading',torch.zeros(48,dtype=torch.long))
    def encode_od(self,x,origin,destination):return (torch.zeros(len(x),64,dtype=D),)
model=TwoRouteModel().eval()
def toy_step(model,q,geo,cell,destination,heading,age,t,hidden,fold,timestamp,event_history):
    p=model.logits.softmax(-1)
    a=torch.zeros(49,dtype=D);a[1]=p[1];a[2]=p[2];a[3]=p[0]
    mid=torch.zeros(49,dtype=D);mid[1]=1
    stay=torch.zeros(49,dtype=D);stay[0]=1
    motion=(cell==0)[:,None]*a[None]+(cell==1)[:,None]*mid[None]+((cell==2)|(cell==3))[:,None]*stay[None]
    return motion,(cell==2).to(D),hidden
B=4096
x=torch.zeros(B,8,dtype=D)
origin=torch.zeros(B,dtype=torch.long);destination=torch.full_like(origin,2)
sample=sample_cohort_trajectories(model,x,origin,destination,particles=8,candidates=16,max_rounds=4,horizon=8,min_steps=1,step_function=toy_step)
assert sample['complete'].all()
times=sample['times']
category=torch.where(times==15,1,2)
assert torch.all((times==15)|(times==30))
expected_logp=model.logits.log_softmax(-1)[category]
assert torch.allclose(sample['log_probability'],expected_logp,atol=1e-12,rtol=0)
truth=torch.full((B,),22.5,dtype=D)
weights,_=conditional_crps_weights(times,truth)
p=model.logits.softmax(-1)
score=torch.nn.functional.one_hot(category,3).to(D)-p.detach()[None,None,:]
per_query=(weights[:,:,None]*score).sum(1)
mean=per_query.mean(0);sem=per_query.std(0)/B**.5
conditional=model.logits[1:].softmax(-1)
support=torch.tensor([15.,30.],dtype=D)
exact=(conditional*(support-22.5).abs()).sum()-.5*(conditional[:,None]*conditional[None,:]*(support[:,None]-support[None,:]).abs()).sum()
g=torch.autograd.grad(exact,model.logits,retain_graph=True)[0]
assert torch.all((mean-g).abs()<5*sem+1e-10),(mean,g,sem)
loss,_=conditional_crps_surrogate(times,truth,sample['log_probability'],scale=1)
autograd=torch.autograd.grad(loss,model.logits)[0]
assert torch.allclose(autograd,mean,atol=1e-10,rtol=0)
assert abs(float(autograd[0]))<1e-12
indicator=(times==15).to(D)
corr=torch.corrcoef(torch.stack([indicator[:,0],indicator[:,1]]))[0,1]
assert abs(float(corr))<.06
assert not torch.cuda.is_initialized()
report=dict(passed=True,synthetic_only=True,complete_queries=B,accepted_samples=int(times.numel()),path_likelihood_matches_unconditional_origin_branch_exact=True,out_of_cohort_trap_rejected=True,no_label_in_sampler_signature=True,exact_gradient=g.detach().numpy(),sample_gradient=mean.detach().numpy(),gradient_standard_error=sem.detach().numpy(),autograd_matches_score_estimator=True,first_two_accepted_draw_correlation=float(corr),no_GPU=True,not_Porto_MAE=True,code_sha256=sha(__file__),sampler_sha256=sha(OUT/'muestreador_rechazo_nativo.py'))
write(OUT/'verificacion_muestreador_rechazo.json',report)
print(json.dumps(report,default=lambda x:x.tolist()))
