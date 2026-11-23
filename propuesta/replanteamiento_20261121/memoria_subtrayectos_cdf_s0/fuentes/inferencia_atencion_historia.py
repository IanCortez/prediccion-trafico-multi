"""Frozen raw64attention inference, retaining only one batch distribution at a time."""
from base_replanteo import *
import torch,importlib,argparse
def load_frozen_attention(run_id,device='cuda'):
 run=(OUT/run_id).resolve();assert run.parent==OUT.resolve();r=read(run/'resultado.json')
 for name,digest in r['sources'].items():assert sha(run/'fuentes'/name)==digest
 assert sha(OUT/'base_replanteo.py')==r['sources']['base_replanteo.py']
 sys.path.insert(0,str(run/'fuentes'))
 helper=importlib.import_module('inferencia_historica_fresca');assert Path(helper.__file__).resolve()==(run/'fuentes/inferencia_historica_fresca.py').resolve()
 parent,base_ck,parent_record,parent_run,prototype_library=helper.load_frozen_history(r['config']['parent_id'],device=device)
 assert parent_record['model_sha256']==r['config']['parent_sha256']
 sys.path.insert(0,str(run/'fuentes'))
 token=importlib.import_module('historia_individual');module=importlib.import_module('modelo_atencion_historia')
 for obj,name in [(token,'historia_individual.py'),(module,'modelo_atencion_historia.py')]:assert Path(obj.__file__).resolve()==(run/'fuentes'/name).resolve()
 path=Path(r['model_file']).resolve();assert path.is_relative_to(run) and sha(path)==r['model_sha256']
 ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['epoch']==r['best_epoch'] and ck['config']==r['config']
 constructor=ck['constructor'];saved=dict(np.load(run/'constructor_tokens.npz'));assert sha(run/'constructor_tokens.npz')==r['config']['normalization_sha256']
 for key in saved:assert np.array_equal(saved[key],constructor[key])
 model=module.IndividualHistoryAttention(parent,constructor['mean'],constructor['std'],constructor['time_anchor'],r['config']['mode'])
 check=model.load_state_dict(ck['branch_state'],strict=False)
 assert not check.unexpected_keys and all(k.startswith('parent.') for k in check.missing_keys)
 model=model.to(device).eval()
 memory=OUT/'datos_prototipos_historicos_s0/memoria_Train.npz';library=token.IndividualHistoryBank(memory,r['config']['memory_sha256'])
 return model,base_ck,r,run,prototype_library,library,helper,token
def raw_inputs(items,base_config,prototype_library,library,helper,token):
 from consulta_od import od_records
 X,C,H,names=helper.raw_inputs(items,base_config,prototype_library)
 d=od_records(items);query=token.query_parts(d);query['indices']=library.query_indices(d,training=False)
 return X,C,H,query,names
@torch.no_grad()
def iter_attention(model,library,X,C,H,query):
 device=next(model.parameters()).device
 for first in range(0,len(X),2048):
  n=min(2048,len(X)-first)
  batch=[torch.as_tensor(z[first:first+n],device=device) for z in [X,C,H]]
  q={key:torch.as_tensor(z[first:first+n],device=device) for key,z in query.items()}
  if n<2048:
   batch=[torch.cat([z,z[-1:].expand((2048-n,)+z.shape[1:])]) for z in batch]
   q={key:torch.cat([z,z[-1:].expand((2048-n,)+z.shape[1:])]) for key,z in q.items()}
  raw=library.features(q['xy_km'],q['hour'],q['weekend'],q['indices'])
  out=model(*batch,raw)
  yield first,n,out
  del out,raw,batch,q
@torch.no_grad()
def predict_quantiles(model,library,X,C,H,query):
 parts=[]
 for first,n,out in iter_attention(model,library,X,C,H,query):
  parts.append(model.duration.quantiles(out[0])[:n].cpu().numpy());del out
 return np.concatenate(parts)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
 access=deny_test();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
 if a.device=='cuda':torch.cuda.set_per_process_memory_fraction(.5)
 source=Path(a.input).resolve();target=Path(a.output).resolve();assert source.is_relative_to(ROOT) and target.is_relative_to(ROOT) and not target.exists()
 model,ck,r,run,proto,library,helper,token=load_frozen_attention(a.run,a.device)
 X,C,H,q,names=raw_inputs(read(source),ck['base_config'],proto,library,helper,token);assert names==ck['features']
 predictions=predict_quantiles(model,library,X,C,H,q)
 write(target,dict(median_seconds=predictions[:,1],q10_seconds=predictions[:,0],q90_seconds=predictions[:,2],model=r['id'],
  uncertainty='Exploratory marginal intervals; empirical coverage needs independent validation.',information=['o_lon','o_lat','d_lon','d_lat','timestamp'],Test_access=False))
if __name__=='__main__':main()
