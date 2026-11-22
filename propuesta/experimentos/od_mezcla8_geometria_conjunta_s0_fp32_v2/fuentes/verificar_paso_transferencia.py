from common import *
import torch,copy
from torch import nn
from paso_transferencia_auditado import audited_transfer_step
class Quadratic(nn.Module):
    def __init__(self):super().__init__();self.theta=nn.Parameter(torch.zeros(2,dtype=torch.float64))
    def loss(self):return .5*(self.theta+1).square().sum()
def setup(momentum=False):
    model=Quadratic();opt=torch.optim.Adam(model.parameters(),lr=.01);opt.state[model.theta]=dict(step=torch.tensor(100.),exp_avg=torch.tensor([-100.,-100.],dtype=torch.float64) if momentum else torch.zeros(2,dtype=torch.float64),exp_avg_sq=torch.tensor([.001,1000.],dtype=torch.float64));return model,opt
model,opt=setup();target=torch.ones(2,dtype=torch.float64);source=torch.tensor([-3.,2.],dtype=torch.float64);projected=source-(source@target)/(target@target)*target;mixed=target+projected;assert mixed@target>0;before=float(model.loss().detach());model.theta.grad=mixed;opt.step();after=float(model.loss().detach());assert after>before
safe,opt=setup();safe.frozen_dropout=nn.Dropout(.5);safe.frozen_dropout.eval();good=torch.ones(2,dtype=torch.float64);result=audited_transfer_step(safe,opt,[target],[[projected],[good]],safe.loss);assert result['accepted'] and result['audit_loss_after']<result['audit_loss_before'];assert result['source_weights'][0]==0 and result['source_weights'][1]>0;assert safe.training and not safe.frozen_dropout.training
rollback,opt=setup(momentum=True);base_model=copy.deepcopy(rollback.state_dict());base_opt=copy.deepcopy(opt.state_dict());no_update=audited_transfer_step(rollback,opt,[target],[[projected]],rollback.loss);assert no_update['status']=='no_update';assert torch.equal(rollback.theta,base_model['theta'])
for key,value in base_opt['state'][0].items():assert torch.equal(value,opt.state_dict()['state'][0][key])
assert opt.param_groups[0]['lr']==base_opt['param_groups'][0]['lr']
# A stochastic evaluation cannot substantiate the deterministic local check.
probe,opt=setup();counter=[0]
def varying():counter[0]+=1;return probe.loss()+counter[0]*.01
rejected=False
try:audited_transfer_step(probe,opt,[target],[],varying)
except AssertionError:rejected=True
assert rejected and torch.equal(probe.theta,torch.zeros(2,dtype=torch.float64))
write_json(BASE/'instrumentos/transferencia_paso_real.json',dict(passed=True,projected_gradient=dict(target_dot_mixed=float(target@mixed),loss_before=before,loss_after_actual_adam=after),audited_update=result,no_update_rollback_passed=True,optimizer_moments_and_step_preserved_when_rejected=True,mixed_submodule_modes_preserved=True,stochastic_audit_rejected=True,true_cross_city_experiment=False,target_training_audit_only=True,test_labels_read=False,code_sha256=sha(BASE/'paso_transferencia_auditado.py')));print(dict(projected_adam_loss_change=after-before,accepted_weights=result['source_weights'],accepted_audit_loss_change=result['audit_loss_after']-result['audit_loss_before'],no_update_rollback=True),flush=True)
