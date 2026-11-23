"""Reproducible inference of a frozen projection mixture from five raw OD fields."""
from base_replanteo import *
import torch,importlib,argparse
def load_frozen(run_id,device='cuda'):
 run=(OUT/run_id).resolve();assert run.parent==OUT.resolve()
 record=read(run/'resultado.json');assert record['new_function_selected']
 checkpoint=Path(record['model_file']);assert checkpoint.resolve().is_relative_to(run)
 assert sha(checkpoint)==record['model_sha256']
 for name,value in record['sources'].items():assert sha(run/'fuentes'/name)==value
 sys.path.insert(0,str(run/'fuentes'))
 from modelo_mezcla_rutas import make_duration_model
 from modelo_mezcla_proyeccion import ProjectionMixture
 ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
 duration=make_duration_model(ck['original_constructor'],ck['parent_config'])
 model=ProjectionMixture(duration,ck['state']['route_mean'].numpy(),ck['state']['route_std'].numpy(),ck['config']['mode'])
 model.load_state_dict(ck['state']);model=model.to(device).eval()
 return model,ck,record,run
def raw_inputs(items,parent_config):
 from consulta_od import od_records,build_od_features
 from rutas_proyeccion_incierta import build_uncertain_routes
 X,names=build_od_features(od_records(items),parent_config)
 candidates,_=build_uncertain_routes(items,training=False)
 return X,candidates.astype(np.float16),names
def iter_outputs(model,X,C):
 device=next(model.parameters()).device;model.eval()
 with torch.no_grad():
  for first in range(0,len(X),2048):
   xx=torch.as_tensor(X[first:first+2048],device=device,dtype=torch.float32)
   cc=torch.as_tensor(C[first:first+2048],device=device,dtype=torch.float16)
   yield first,model(xx,cc)
def predict_quantiles(model,X,C):
 return np.concatenate([model.duration.quantiles(out[0]).cpu().numpy() for first,out in iter_outputs(model,X,C)])
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--device',choices=['cuda','cpu'],default='cuda');a=p.parse_args()
 access=deny_test();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
 if a.device=='cuda':torch.cuda.set_per_process_memory_fraction(.4)
 source=Path(a.input).resolve();dest=Path(a.output).resolve()
 assert source.is_relative_to(ROOT.resolve()) and dest.is_relative_to(ROOT.resolve()) and not dest.exists()
 items=read(source);assert isinstance(items,list) and len(items)>0
 model,ck,r,run=load_frozen(a.run,a.device);X,C,names=raw_inputs(items,ck['parent_config']);assert names==ck['features']
 q=predict_quantiles(model,X,C)
 write(dest,dict(model_sha256=r['model_sha256'],input_fields=['o_lon','o_lat','d_lon','d_lat','timestamp'],
  predictions=[dict(input_index=i,q10_s=float(v[0]),median_s=float(v[1]),q90_s=float(v[2])) for i,v in enumerate(q)],Test_access=False))
 print(json.dumps(dict(n=len(q),output=str(dest),model_sha256=r['model_sha256'],Test_access=False)))
if __name__=='__main__':main()
