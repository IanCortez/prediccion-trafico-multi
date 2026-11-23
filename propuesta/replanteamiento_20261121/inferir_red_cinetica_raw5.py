"""Reproduciblefinalproposal inference from raw5 only; allread/write paths confinedtoEproject."""
from base_replanteo import *
import torch,gc
from operador_red_cinetica import RoadClockKernel,RoadStateGeometry,load_counts
from operador_red_cinetica_GPU import CudaRoadClock
from modelo_compuerta_red_cinetica import RoadCDFGate,canonical_gate,point_quantiles
from modelo_control_lognormal_puente import PublicLognormalControl,canonical_predict_control
from datos_mezcla_individual_Train import build_public206
from coordenadas_trayecto_2d import metric_queries
def predict(items,kind='kinetic'):
 assert kind in ['position','kinetic'] and len(items)>0
 assert all(set(item)=={'o_lon','o_lat','d_lon','d_lat','timestamp'} for item in items)
 query={key:np.asarray([item[key] for item in items]) for key in items[0]};metric_queries(query)
 run=OUT/'red_cinetica_Train_s0';result=read(run/'resultado.json');assert result['complete']
 countsfile=OUT/'conteos_cineticos_prefijo_Train_s0/conteos_fuentes.npz';assert sha(countsfile)==result['source_count_sha256']
 raw,names=build_public206(items,read(BEST/'resultado.json')['config']);assert raw.shape==(len(items),206)
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(.5);torch.use_deterministic_algorithms(False)
 reference=read(OUT/'referencia_publica_Train_red_vial.json');assert sha(Path(reference['model_file']))==reference['model_sha256']
 ck=torch.load(reference['model_file'],map_location='cpu',weights_only=False);baseline=PublicLognormalControl(**ck['constructor']).cuda();baseline.load_state_dict(ck['state'])
 x=torch.as_tensor(raw,device='cuda');lp0,_=canonical_predict_control(baseline,x);del baseline,ck;gc.collect();torch.cuda.empty_cache()
 counts=load_counts(countsfile);geometry=RoadStateGeometry(DATA/'mapa_dirigido_2014.npz')
 torch.use_deterministic_algorithms(True);operator=CudaRoadClock(RoadClockKernel(counts,geometry,kind=kind,excluded_fold=5))
 generated=operator.predict(query);del operator,counts;gc.collect();torch.cuda.empty_cache()
 torch.use_deterministic_algorithms(False)
 checkpoint=Path(result['weights'][kind]['file']);assert checkpoint.resolve().is_relative_to(run.resolve()) and sha(checkpoint)==result['weights'][kind]['sha256']
 state=torch.load(checkpoint,map_location='cpu',weights_only=False);gate=RoadCDFGate(**state['constructor']).cuda();gate.load_state_dict(state['state'])
 supported=torch.as_tensor(generated['supported'],device='cuda');graph=torch.as_tensor(generated['probability'].astype(np.float32),device='cuda',dtype=torch.float64)
 graph=torch.where(supported[:,None],graph/graph.sum(1).clamp_min(1e-300)[:,None],torch.zeros_like(graph))
 lp,g=canonical_gate(gate,x,lp0.cuda(),graph,supported);q=point_quantiles(lp)
 return dict(pred=q[:,1].numpy(),quantiles=q.numpy(),log_prob=lp.numpy(),gate=g.numpy(),graph_supported=generated['supported'])
def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--queries',required=True);parser.add_argument('--output',required=True);parser.add_argument('--arm',choices=['position','kinetic'],default='kinetic')
 args=parser.parse_args();queryfile=Path(args.queries).resolve();output=Path(args.output).resolve()
 assert queryfile.is_relative_to(ROOT.resolve()) and output.is_relative_to(ROOT.resolve()) and not output.exists()
 access=deny_test()
 forbidden=(OUT/'memoria_puntos_Train_s0/puntos_Train.npz').resolve()
 def guard(event,a):
  if event=='open' and a and isinstance(a[0],(str,bytes,os.PathLike)):
   path=Path(os.fsdecode(a[0])).resolve()
   assert path!=forbidden,'Raw5inference cannot open originalGPSpointbank'
   assert not path.name.startswith(('train_y','val_y','test')),'No evaluationlabels or Test input during raw5prediction'
 sys.addaudithook(guard)
 result=predict(read(queryfile),args.arm);np.savez_compressed(output,**result)
 print(json.dumps(dict(complete=True,n=len(result['pred']),output=str(output),scope='FinalTrainpilot model, not primarybest or confirmation.',arm=args.arm)))
if __name__=='__main__':main()
