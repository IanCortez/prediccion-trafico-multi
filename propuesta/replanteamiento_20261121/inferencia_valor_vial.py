"""Frozen street-value inference from only five raw OD/departure fields."""
from base_replanteo import *
import torch,argparse,importlib.util
def load_frozen_street(run_id,device='cuda'):
 run=(OUT/run_id).resolve();assert run.parent==OUT.resolve();r=read(run/'resultado.json');assert r['model_file']
 path=Path(r['model_file']).resolve();assert path.is_relative_to(run) and sha(path)==r['model_sha256']
 for name,value in r['sources'].items():assert sha(run/'fuentes'/name)==value
 spec=importlib.util.spec_from_file_location('frozen_street_value_model',run/'fuentes/modelo_valor_vial.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 state=torch.load(path,map_location='cpu',weights_only=False);folder=OUT/'banco_valor_vial_s0';meta=read(folder/'resultado.json')
 assert sha(folder/'grafo_estatico.npz')==meta['static_graph_sha256'];assert sha(run/'normalizacion_Train.npz')==r['normalization_sha256']
 g=dict(np.load(folder/'grafo_estatico.npz'));norm=dict(np.load(run/'normalizacion_Train.npz'))
 model=module.RoadValueNet(norm['mean'],norm['std'],g['static'],g['neighbors'],g['neighbor_valid']);model.load_state_dict(state['state']);model=model.to(device).eval()
 return model,state,r,run
def raw_inputs(items):
 from consulta_od import od_records,build_od_features
 from proyecciones_valor_vial import project_street_query
 d=od_records(items);X,names=build_od_features(d,read(BEST/'resultado.json')['config']);bank=project_street_query(d)
 return X,bank,names
def iter_street(model,X,bank):
 device=next(model.parameters()).device;model.eval()
 with torch.no_grad():
  for first in range(0,len(X),2048):
   end=min(first+2048,len(X));n=end-first;xx=torch.as_tensor(X[first:end],device=device,dtype=torch.float32)
   bb={key:torch.as_tensor(value[first:end].astype(np.int64) if key=='chain' else value[first:end],device=device) for key,value in bank.items()}
   if n<2048:
    xx=torch.cat([xx,xx[-1:].expand(2048-n,-1)])
    bb={key:torch.cat([value,value[-1:].expand((2048-n,)+value.shape[1:])]) for key,value in bb.items()}
   logp=model.public_logprob(xx,**bb);yield first,n,logp
def predict_quantiles(model,X,bank):return np.concatenate([model.quantiles(lp)[:n].cpu().numpy() for first,n,lp in iter_street(model,X,bank)])
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--device',choices=['cuda','cpu'],default='cuda');a=p.parse_args()
 deny_test();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
 if a.device=='cuda':torch.cuda.set_per_process_memory_fraction(.5)
 src=Path(a.input).resolve();dst=Path(a.output).resolve();assert src.is_relative_to(ROOT.resolve()) and dst.is_relative_to(ROOT.resolve()) and not dst.exists()
 model,ck,r,run=load_frozen_street(a.run,a.device);X,b,names=raw_inputs(read(src));assert names==ck['features'];q=predict_quantiles(model,X,b)
 write(dst,dict(model_sha256=r['model_sha256'],input_fields=['o_lon','o_lat','d_lon','d_lat','timestamp'],predictions=[dict(index=i,q10_s=float(v[0]),median_s=float(v[1]),q90_s=float(v[2])) for i,v in enumerate(q)],Test_access=False))
 print(json.dumps(dict(n=len(q),output=str(dst),Test_access=False)),flush=True)
if __name__=='__main__':main()
