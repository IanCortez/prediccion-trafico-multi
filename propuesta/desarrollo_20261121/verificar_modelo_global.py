"""Exact prepared/raw OD replay of the fresh global-route model; no training or Test."""
from piloto_transferencia import ROOT,BASE,OUT,FROZEN,write,sha
from inferencia import feature_input
from consulta_od import od_records,build_od_features
from generar_rutas import route_batch
from modelo_mezcla_conjunta import JointRouteTimeMixture
from scipy.spatial import cKDTree
from numba import set_num_threads
import numpy as np,pandas as pd,torch,json,time,socket

def main():
 start=time.time();set_num_threads(4);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 def deny(*args,**kw):raise RuntimeError('Offline replay')
 socket.socket.connect=deny;socket.create_connection=deny
 run=OUT/'modelo_global_conjunto_s0';r=json.loads((run/'resultado.json').read_text());assert sha(run/'modelo.pt')==r['model_sha256'];rec=json.loads((BASE/'experimentos/od_mezcla8_geometria_conjunta_s0_fp32_v2/resultado.json').read_text());ck=torch.load(run/'modelo.pt',map_location='cpu',weights_only=False);m=JointRouteTimeMixture(ck['mean'],ck['std'],ck['keep'],ck['bins'],ck['anchors'],ck['anchor_indices'],ck['geometry_indices'],16,512).cuda();m.load_state_dict(ck['state']);m.eval()
 @torch.no_grad()
 def predict(X):
  out=[]
  for first in range(0,len(X),2048):
   x=torch.tensor(X[first:first+2048],device='cuda');n=len(x)
   if n<2048:x=torch.cat([x,x[-1:].expand(2048-n,-1)],0)
   out.append(m.quantiles(m(x)[0],(.5,))[:n,0].cpu().numpy())
  return np.concatenate(out)
 d,X,names=feature_input('validation',rec);z=np.load(OUT/'rutas_globales_completas/validation_rutas_globales.npz');assert np.array_equal(z['row'],d['row']);names+=z['names'].tolist();X=np.column_stack([X,z['X']]);assert names==ck['features'];p=predict(X);saved=np.load(run/'validacion.npz');assert np.array_equal(saved['row'],d['row']) and np.array_equal(p,saved['pred'])
 raw=dict(np.load(BASE/'datos/mapa_dirigido_2014.npz'));order=np.argsort(raw['u'],kind='stable').astype(np.int32);ptr=np.r_[0,np.cumsum(np.bincount(raw['u'],minlength=len(raw['xy'])))].astype(np.int64);dest=raw['v'][order];tree=cKDTree(raw['xy']);times=np.load(BASE/'datos/perfiles_mapmatched_2014/time_s.npy',mmap_mode='r');support=np.load(BASE/'datos/perfiles_mapmatched_2014/link_support.npy',mmap_mode='r')
 def raw_X(items):
  q=od_records(items);original,onames=build_od_features(q,rec['config']);origin=np.column_stack([(q['o_lon']+8.62)*111195*np.cos(np.deg2rad(41.16)),(q['o_lat']-41.16)*111195]);end=np.column_stack([(q['d_lon']+8.62)*111195*np.cos(np.deg2rad(41.16)),(q['d_lat']-41.16)*111195]);_,src=tree.query(origin);_,tar=tree.query(end);dt=pd.to_datetime(q['timestamp'],unit='s',utc=True).tz_convert('Europe/Lisbon');profile=(840+np.asarray(dt.dayofweek)*24+np.asarray(dt.hour)).astype(np.int32);new=route_batch(origin,end,src,tar,profile,raw['xy'],ptr,dest,order,raw['length'],raw['seconds'],times,support);return np.column_stack([original,new])
 fields=['o_lon','o_lat','d_lon','d_lat','timestamp'];items=[{k:d[k][i].item() for k in fields} for i in range(65)];f=raw_X(items);pr=predict(f);rev=predict(raw_X(items[::-1]))[::-1];single=predict(raw_X(items[:1]));assert np.array_equal(pr,p[:65]) and np.array_equal(rev,pr) and np.array_equal(single,pr[:1])
 proof=dict(passed=True,model_sha256=r['model_sha256'],validation_n=len(p),validation_prediction_bit_identical=True,raw65_bit_identical=True,reverse_bit_identical=True,single_bit_identical=True,query_information=fields,Test_access=False,seconds=time.time()-start,code_sha256=sha(ROOT.__class__(__file__)))
 write(run/'verificacion_inferencia.json',proof);print(json.dumps(proof),flush=True)
if __name__=='__main__':main()
