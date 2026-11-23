"""Verify any promising fixed-encoder semantic candidate from raw five-field queries."""
from base_replanteo import *
import torch,shutil
from inferencia import feature_input
from consulta_od import od_records,build_od_features
from modelo_mezcla_rutas import make_duration_model
from modelo_tipo_latente import LatentTypeTilt,duration_median
access=deny_test();torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
run=OUT/'tipo_latente_cpu_s0';r=read(run/'resultado.json')
assert r['qualified_candidates'],'Do not spend reproduction work on discarded heads'
target=run/'verificacion_consulta_cruda.json';assert not target.exists()
start=time.time();rec=read(BEST/'resultado.json')
assert sha(BEST/'modelo.pt')==r['parent_sha256']
for f,h in r['sources'].items():assert sha(run/'fuentes'/f)==h
for name,item in r['outputs'].items():assert sha(run/(name+'.pt'))==item['model_sha256']
ck=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False)
parent=make_duration_model(ck,rec['config']);parent.load_state_dict(ck['state']);parent.eval()
heads={}
for item in r['qualified_candidates']:
 name=item['name'];saved=torch.load(run/(name+'.pt'),map_location='cpu',weights_only=False)
 state=saved['state'];head=LatentTypeTilt(state['mean'],state['std']);head.load_state_dict(state);heads[name]=head.eval()
va,V,names=feature_input('validation',rec);assert names==rec['features']
ix=np.sort(np.random.default_rng(20261122).choice(len(V),65,replace=False))
cols=['o_lon','o_lat','d_lon','d_lat','timestamp']
items=[{k:int(va[k][i]) if k=='timestamp' else float(va[k][i]) for k in cols} for i in ix]
raw=od_records(items)
R,n=build_od_features(raw,rec['config']);assert n==names
maxerr=float(np.abs(R-V[ix]).max())
latest=[]
hook=parent.backbone.output.register_forward_pre_hook(lambda module,args:latest.append(args[0].detach().mean(1)))
def predict(F,batch=32):
 result={name:[] for name in heads}
 with torch.no_grad():
  for at in range(0,len(F),batch):
   out=parent(torch.from_numpy(F[at:at+batch]));h=latest.pop()
   p=out[0].exp().mean(1);p=p/p.sum(-1,keepdim=True)
   for name,head in heads.items():result[name].append(duration_median(head(h,p)[0]).numpy())
 return {name:np.concatenate(q) for name,q in result.items()}
baseline=predict(V[ix]);raw_pred=predict(R);reverse=predict(R[::-1].copy())
single=predict(R[:4],batch=1);stored=dict(np.load(run/'validacion.npz'))
checks={}
for name in heads:
 checks[name]=dict(raw_equals_saved=bool(np.array_equal(raw_pred[name],stored[name][ix])),precomputed_equals_saved=bool(np.array_equal(baseline[name],stored[name][ix])),reverse_equal=bool(np.array_equal(reverse[name][::-1],raw_pred[name])),singleton_equal=bool(np.array_equal(single[name],raw_pred[name][:4])))
# Explicitly ensure observed call is rejected at the public query boundary.
forbidden_rejected=False
try:
 bad=[dict(items[0],call='A')]
 od_records(bad)
except AssertionError:forbidden_rejected=True
assert forbidden_rejected
hook.remove()
proof=dict(passed=all(all(q.values()) for q in checks.values()),checks=checks,query_fields=cols,observed_call_rejected=True,raw_feature_max_error=maxerr,raw_n=len(ix),Test_access=False,actual_files_accessed=sorted(access),code_sha256=sha(__file__),source_sha256=sha(OUT/'modelo_tipo_latente.py'),parent_sha256=r['parent_sha256'],CPU_canonical_batch=32,seconds=time.time()-start)
write(target,proof);shutil.copy2(__file__,run/'fuentes'/Path(__file__).name)
if proof['passed'] and r['reference_CPU_equal_protected']:
 for item in r['qualified_candidates']:item['status']='qualified_full_Validation_and_raw_CPU_reproduced'
 r['raw_verification_file']=target.name;r['raw_verification_passed']=True
 write(run/'resultado.json',r)
else:
 r['raw_verification_file']=target.name;r['raw_verification_passed']=proof['passed'];write(run/'resultado.json',r)
print(json.dumps(proof),flush=True)
