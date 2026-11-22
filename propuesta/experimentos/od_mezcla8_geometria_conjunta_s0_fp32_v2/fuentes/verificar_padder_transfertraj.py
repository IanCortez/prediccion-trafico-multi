from common import *
import ast,pandas as pd,torch
source=BASE/'referencias/TransferTraj/data.py';tree=ast.parse(source.read_text(encoding='utf-8-sig'));names={'X_COL','Y_COL','T_COL','DT_COL','X_norm_COL','Y_norm_COL','ST_MAP','FEATURE_PAD','KNOWN_TOKEN','MASK_TOKEN','UNKNOWN_TOKEN','PAD_TOKEN'};env=dict(np=np,torch=torch)
for node in tree.body:
    if isinstance(node,ast.Assign):
        for target in node.targets:
            if isinstance(target,ast.Name) and target.id in names:env[target.id]=ast.literal_eval(node.value)
definitions=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in ('OdTtePadder','pad_batch_2d','pad_batch_3d')];assert len(definitions)==3
exec(compile(ast.Module(body=definitions,type_ignores=[]),str(source),'exec'),env);padder=env['OdTtePadder']()
def frame(n,arrival,perturb=False):
    x=np.linspace(0,1000,n);y=np.linspace(0,500,n)
    if perturb:x[1:-1]+=np.sin(np.arange(n-2))*400;y[1:-1]+=300
    delta=np.linspace(0,arrival,n);return pd.DataFrame(dict(lng=x,lat=y,timestamp=1399000000+delta,delta_t=delta,first_lng=500000+x,first_lat=4550000+y))
base=padder([frame(12,165)]);cases={}
for name,traj in [('hidden_arrival',frame(12,1200)),('hidden_interior_route',frame(12,165,True)),('hidden_point_count',frame(91,1350,True))]:
    changed=padder([traj]);same=[torch.equal(base[index],changed[index]) for index in (0,2,3)];assert all(same),(name,same)
    cases[name]=dict(input_identical=True,positions_identical=True,origin_reference_identical=True,target_changed=not torch.equal(base[1],changed[1]))
assert base[0].shape[1]==2 and torch.all(base[0][0,1,2:,0]==0)
write_json(BASE/'instrumentos/transfertraj_padder_od.json',dict(passed=True,source_sha256=sha(source),pinned_commit='26d642e05e0afc75b9166a6bc73ed3876e9f4cf9',cases=cases,input_shape=list(base[0].shape),test_labels_read=False,scope='Official public OD collator, isolated without running preprocessing or training. This verifies invariance at its output, not the unavailable end-to-end published Porto evaluation or metric conversion.'))
print(cases,flush=True)
