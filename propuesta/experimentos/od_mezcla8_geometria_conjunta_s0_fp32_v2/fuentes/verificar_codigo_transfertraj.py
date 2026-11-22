from common import *
import ast,torch
from torch import nn
from torch.nn import functional as F
from types import SimpleNamespace
folder=BASE/'referencias/TransferTraj';source=folder/'models/transfertraj.py';tree=ast.parse(source.read_text(encoding='utf-8-sig'));cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TransferTraj');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='cal_modal_h');selected=[n for n in method.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('poi_distance','poi_mask') for t in n.targets)];assert len(selected)==2
positions=torch.tensor([[0.,0.],[5.,0.],[10.,0.],[15.,0.],[90.,0.],[100.,0.],[110.,0.]]);scope=dict(torch=torch,self=SimpleNamespace(poi_coors=positions,poi_dist=100),spatial=torch.zeros((1,1,2)),first_point=torch.zeros((1,2)));exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),scope);included=scope['poi_mask'].flatten().tolist();assert included==[True,True,False,False,False,False,False]
enc=folder/'models/encode.py';tree=ast.parse(enc.read_text(encoding='utf-8-sig'));router=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NoisyTopkRouter');scope=dict(torch=torch,nn=nn,F=F);exec(compile(ast.Module(body=[router],type_ignores=[]),str(enc),'exec'),scope);torch.manual_seed(1729);m=scope['NoisyTopkRouter'](16,8,4).eval();x=torch.zeros((64,16))
with torch.no_grad():a,ia,_=m(x);b,ib,_=m(x)
changed=float(torch.any(ia!=ib,dim=-1).float().mean());assert changed>0;max_difference=float(torch.max(torch.abs(a-b)))
pipeline=folder/'pipeline.py';tree=ast.parse(pipeline.read_text(encoding='utf-8-sig'));aliases=[]
for function in tree.body:
    if isinstance(function,ast.FunctionDef) and function.name in ('train_model','finetune_model'):
        for node in ast.walk(function):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='saved_model_state_dict' for t in node.targets) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='state_dict':aliases.append(function.name)
probe=nn.Linear(1,1,bias=False);state=probe.state_dict();before=state['weight'].clone()
with torch.no_grad():probe.weight.add_(1.)
alias_changed=bool(torch.allclose(state['weight'],before+1));assert alias_changed and len(aliases)==2
out=dict(commit='26d642e05e0afc75b9166a6bc73ed3876e9f4cf9',radius_check=dict(poi_parameter=100,distances_m=positions[:,0].tolist(),included=included,effective_radius_m=10.,note='Published code compares squared UTM distances to the parameter. It is a squared-distance threshold, not a100m radius when configured100.'),eval_router_check=dict(mode='eval',identical_inputs=True,different_topk_fraction=changed,max_gate_difference=max_difference,note='Router draws random noise without a training-mode guard. This synthetic check does not quantify prediction or MAE variation.'),checkpoint_check=dict(functions_assigning_live_state_dict=aliases,stored_tensor_changed_after_parameter_update=alias_changed,note='state_dict tensors share storage unless cloned/deep-copied. No validation selection is implemented in these training functions.'),no_query_label_leakage_claim=True,published_mae_not_invalidated_by_these_unit_tests=True,test_labels_read=False,sources={str(p.relative_to(folder)):sha(p) for p in (source,enc,pipeline)})
write_json(BASE/'instrumentos/transfertraj_codigo_auditoria.json',out);print(json.dumps(out),flush=True)
new_files=('pipeline.py','utils.py','models/encode.py');write_json(folder/'adiciones_codigo.json',dict(commit=out['commit'],files={name:dict(sha256=sha(folder/name),url='https://github.com/wtl52656/TransferTraj/blob/'+out['commit']+'/'+name) for name in new_files}))
