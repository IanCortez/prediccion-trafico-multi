"""Point prediction from five fields using frozen explicit median artifacts."""
from base_replanteo import *
import torch,importlib,argparse
def load_frozen_direct(run_id,device='cuda'):
 run=(OUT/run_id).resolve();assert run.parent==OUT.resolve();r=read(run/'resultado.json')
 for name,digest in r['sources'].items():assert sha(run/'fuentes'/name)==digest
 sys.path.insert(0,str(run/'fuentes'))
 module=importlib.import_module('modelo_mediana_directa')
 assert Path(module.__file__).resolve()==(run/'fuentes/modelo_mediana_directa.py').resolve()
 helper=importlib.import_module('inferencia_historica_fresca')
 assert Path(helper.__file__).resolve()==(run/'fuentes/inferencia_historica_fresca.py').resolve()
 reference,base_ck,base_record,base_run,library=helper.load_frozen_history(r['config']['reference_id'],device=device)
 assert base_record['model_sha256']==r['config']['reference_sha256'] and base_record['model_file']==r['config']['reference_model_file']
 torch.manual_seed(r['config']['seed']);model=module.DirectMedianReadout(reference)
 if r['best_epoch']>0:
  file=Path(r['model_file']).resolve();assert file.is_relative_to(run) and sha(file)==r['model_sha256']
  ck=torch.load(file,map_location='cpu',weights_only=False)
  assert ck['reference_sha256']==r['config']['reference_sha256'] and ck['epoch']==r['best_epoch']
  model.student.load_state_dict(ck['student_state']);model.readout.load_state_dict(ck['readout_state'])
 else:assert r['model_file'] is None
 model=model.to(device).eval();model.set_student_training(False)
 return model,base_ck,r,run,library,helper
@torch.no_grad()
def predict_points(model,X,C,H):
 device=next(model.parameters()).device;points=[];continuous=[]
 for first in range(0,len(X),2048):
  batch=[torch.as_tensor(z[first:first+2048],device=device) for z in [X,C,H]];n=len(batch[0])
  if n<2048:batch=[torch.cat([z,z[-1:].expand((2048-n,)+z.shape[1:])]) for z in batch]
  members=model(*batch);mean=members.mean(1)
  points.append(model.quantize(mean)[:n].cpu().numpy());continuous.append(mean[:n].cpu().numpy())
 return np.concatenate(points),np.concatenate(continuous)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
 access=deny_test();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
 if a.device=='cuda':torch.cuda.set_per_process_memory_fraction(.4)
 source=Path(a.input).resolve();target=Path(a.output).resolve()
 assert source.is_relative_to(ROOT) and target.is_relative_to(ROOT) and not target.exists()
 model,ck,r,run,library,helper=load_frozen_direct(a.run,a.device)
 items=read(source);assert isinstance(items,list) and len(items)>0
 X,C,H,names=helper.raw_inputs(items,ck['base_config'],library);assert names==ck['features']
 point,continuous=predict_points(model,X,C,H)
 write(target,dict(predictions_seconds=point,continuous_point_seconds=continuous,model=r['id'],
  information=['o_lon','o_lat','d_lon','d_lat','timestamp'],uncertainty='Point readout; no new interval calibration claim.',Test_access=False))
if __name__=='__main__':main()
