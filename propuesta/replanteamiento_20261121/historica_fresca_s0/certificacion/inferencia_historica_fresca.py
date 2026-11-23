"""Inference of a frozen fresh historical route model from five OD/departure fields."""
from base_replanteo import *
import torch,importlib,argparse
def load_frozen_history(run_id,device='cuda'):
 run=(OUT/run_id).resolve();assert run.parent==OUT.resolve();record=read(run/'resultado.json')
 checkpoint=Path(record['model_file']).resolve();assert checkpoint.is_relative_to(run) and sha(checkpoint)==record['model_sha256']
 for name,value in record['sources'].items():assert sha(run/'fuentes'/name)==value
 sys.path.insert(0,str(run/'fuentes'))
 from modelo_mezcla_rutas import make_duration_model
 projection=importlib.import_module('modelo_mezcla_proyeccion');historical=importlib.import_module('modelo_mezcla_historica')
 assert Path(projection.__file__).resolve()==(run/'fuentes/modelo_mezcla_proyeccion.py').resolve()
 assert Path(historical.__file__).resolve()==(run/'fuentes/modelo_mezcla_historica.py').resolve()
 ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
 duration=make_duration_model(ck['base_constructor'],ck['base_config'])
 mc=ck['map_constructor'];hc=ck['history_constructor'];parent=projection.ProjectionMixture(duration,mc['mean'],mc['std'],'expanded')
 model=historical.HistoricalRouteMixture(parent,hc['mean'],hc['std'],hc['time_anchor'],ck['config']['mode'])
 model.load_state_dict(ck['state']);model=model.to(device).eval()
 data=OUT/'datos_prototipos_historicos_s0';meta=read(data/'resultado.json')
 memory=data/'memoria_Train.npz';assert sha(memory)==meta['memory']['sha256']
 hist=importlib.import_module('historicos_rutas');assert Path(hist.__file__).resolve()==(run/'fuentes/historicos_rutas.py').resolve()
 library=hist.HistoryRouteBank(memory)
 return model,ck,record,run,library
def raw_inputs(items,base_config,library):
 from consulta_od import od_records,build_od_features
 from rutas_proyeccion_incierta import build_uncertain_routes
 d=od_records(items);X,names=build_od_features(d,base_config)
 C,_=build_uncertain_routes(items,training=False);C=C.astype(np.float16)
 H,source_rows=library._build(d,C,training=False)
 return X,C,H,names
def iter_outputs(model,X,C,H):
 device=next(model.parameters()).device;model.eval()
 with torch.no_grad():
  for first in range(0,len(X),2048):
   x=torch.as_tensor(X[first:first+2048],device=device,dtype=torch.float32)
   c=torch.as_tensor(C[first:first+2048],device=device,dtype=torch.float16)
   h=torch.as_tensor(H[first:first+2048],device=device,dtype=torch.float16)
   yield first,model(x,c,h)
def predict_quantiles(model,X,C,H):
 return np.concatenate([model.duration.quantiles(out[0]).cpu().numpy() for _,out in iter_outputs(model,X,C,H)])
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--device',choices=['cuda','cpu'],default='cuda');a=p.parse_args()
 deny_test();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
 if a.device=='cuda':torch.cuda.set_per_process_memory_fraction(.4)
 src=Path(a.input).resolve();dest=Path(a.output).resolve()
 assert src.is_relative_to(ROOT.resolve()) and dest.is_relative_to(ROOT.resolve()) and not dest.exists()
 items=read(src);assert isinstance(items,list) and len(items)>0
 model,ck,r,run,library=load_frozen_history(a.run,a.device)
 X,C,H,names=raw_inputs(items,ck['base_config'],library);assert names==ck['features'];q=predict_quantiles(model,X,C,H)
 write(dest,dict(model_sha256=r['model_sha256'],input_fields=['o_lon','o_lat','d_lon','d_lat','timestamp'],
  predictions=[dict(input_index=i,q10_s=float(v[0]),median_s=float(v[1]),q90_s=float(v[2])) for i,v in enumerate(q)],Test_access=False))
 print(json.dumps(dict(n=len(q),output=str(dest),Test_access=False)))
if __name__=='__main__':main()
