"""Check numerical median sensitivity of reference renormalization, CPU only."""
from base_replanteo import *
from modelo_mezcla_rutas import make_duration_model
from inferencia import feature_input
import torch
torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
access=deny_test()
start=time.time()
rec=read(BEST/'resultado.json')
ck=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False)
model=make_duration_model(ck,rec['config']).eval();model.load_state_dict(ck['state'])
va,V,names=feature_input('validation',rec)
old=np.load(BEST/'validacion.npz')
original=[];normalized=[];near=[]
grid=torch.arange(60,1786,15)
with torch.no_grad():
    for first in range(0,len(V),2048):
        x=torch.tensor(V[first:first+2048]);n=len(x)
        if n<2048:x=torch.cat([x,x[-1:].expand(2048-n,-1)])
        p=model(x)[0].exp().mean(1)
        med=lambda q:grid[(q.cumsum(-1)<.5).sum(-1).clamp_max(115)]
        a=med(p)[:n].numpy();b=med(p/p.sum(-1,keepdim=True))[:n].numpy()
        original.append(a);normalized.append(b)
        distance=(p.cumsum(-1)-.5).abs().min(-1).values[:n]
        nn=np.flatnonzero(distance.numpy()<2e-6)+first
        near.extend(nn.tolist())
a=np.concatenate(original);b=np.concatenate(normalized)
changed=np.flatnonzero(a!=b)
assert not torch.cuda.is_initialized()
report=dict(n=len(a),CPU_reference_vs_retained_point_changes=int((a!=old['pred']).sum()),CPU_renormalization_changes=len(changed),changed_validation_indices=changed.tolist(),near_half_indices=near,numerical_scope='CPU FP32 highest canonical2048; potential sensitivity only. GPU inference remains a separate exact reference check.',no_GPU=True,Test_access=False,actual_files_accessed=sorted(access),code_sha256=sha(__file__),seconds=time.time()-start)
write(OUT/'verificacion_normalizacion_referencia_cpu.json',report)
print(json.dumps(report))
