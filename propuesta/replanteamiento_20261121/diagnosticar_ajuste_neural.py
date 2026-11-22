"""CPU diagnostic of a stable in-progress checkpoint on actual TRAIN teaching states only."""
from base_replanteo import *
from modelo_direccional import DirectionalArrival,heading_ids
from inferencia import feature_input
import torch,io
torch.set_num_threads(4);access=deny_test();run=OUT/'espacial_direccional_s0';p=run/'modelo.pt'
for attempt in range(5):
 try:
  before=read(run/'mejor.json');stat=p.stat();blob=p.read_bytes();after=read(run/'mejor.json');newstat=p.stat()
  if before!=after or stat.st_mtime_ns!=newstat.st_mtime_ns or len(blob)!=stat.st_size:time.sleep(.2);continue
  ck=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False);break
 except (RuntimeError,OSError,EOFError):time.sleep(.2)
else:raise RuntimeError('No stable checkpoint snapshot')
grid=read(OUT/'datos_semi_markov/protocolo.json');graph=dict(np.load(OUT/'datos_semi_markov/grafo_grid.npz'));distance=np.load(OUT/'datos_semi_markov/distancias_grid.npy');m=DirectionalArrival(ck['stats'],grid,graph,distance,ck['power']).eval();m.load_state_dict(ck['state']);rec=read(BEST/'resultado.json');tr,X,names=feature_input('train',rec);assert names==ck['features'];data=np.load(OUT/'datos_semi_markov/runs_train.npz');cell=data['cell'].astype(np.int32);off=data['run_offsets'];trip=data['trip_train_index'];valid=np.unpackbits(data['outgoing_valid_bits'])[:len(cell)].astype(bool);ri=np.random.default_rng(2026).choice(len(cell)-1,80000,replace=False);ti=np.searchsorted(off,ri,side='right')-1;ix=trip[ti];ok=valid[ri]&(ri<off[ti+1]-1)&(cell[ri]<m.N)&(cell[ri+1]<m.N)&((tr['timestamp'][ix]//(7*86400))%5==0);ri=ri[ok][:8192];ti=ti[ok][:8192];ix=ix[ok][:8192];assert len(ri)==8192
def ids(tag):
 xx=(tr[tag+'_lon']+8.62)*111195*np.cos(np.deg2rad(41.16));yy=(tr[tag+'_lat']-41.16)*111195;return np.floor((xx-grid['x0'])/200).astype(np.int64)*grid['ny']+np.floor((yy-grid['y0'])/200).astype(np.int64)
origin=ids('o');dest=ids('d');nll=[];wnll=[];acc=[]
with torch.no_grad():
 for first in range(0,len(ri),512):
  rr=ri[first:first+512];ii=ix[first:first+512];tt=ti[first:first+512];cur=torch.tensor(cell[rr],dtype=torch.long);nxt=torch.tensor(cell[rr+1],dtype=torch.long);prev=torch.tensor(cell[np.maximum(rr-1,0)],dtype=torch.long);has=torch.tensor((rr>off[tt])&(cell[np.maximum(rr-1,0)]<m.N)&valid[np.maximum(rr-1,0)]);hh=torch.where(has,heading_ids(cur//grid['ny']-prev//grid['ny'],cur%grid['ny']-prev%grid['ny']),4);x=torch.tensor(X[ii]);q,oo,dd,p0,e,we=m.encode_od(x,torch.tensor(origin[ii]),torch.tensor(dest[ii]));pp,ww=m.state_parameters(q,x[:,:4],cur,dd,hh);dx=nxt//grid['ny']-cur//grid['ny'];dy=nxt%grid['ny']-cur%grid['ny'];code=(dx+3)*7+dy+3;act=code-(code>24).long();assert torch.all(m.allowed[cur,act]);chosen=pp.gather(1,act[:,None])[:,0];d=torch.tensor(data['duration_n'][rr].astype(np.int64))-1;nll.extend((-chosen.clamp_min(1e-9).log()).numpy());wnll.extend((-ww.gather(1,d[:,None])[:,0].clamp_min(1e-9).log()).numpy());acc.extend((pp.argmax(-1)==act).numpy())
report=dict(scope='8192 random valid nonterminal TRAIN teaching states, weeks modulo5==0; model itself trained on ALL eligible Train including these weeks',independent_generalization_test=False,optimistic_fit_diagnostic=True,snapshot_stage=before['stage'],snapshot_sha256=hashlib.sha256(blob).hexdigest(),next_move_NLL_nats=float(np.mean(nll)),next_move_top1=float(np.mean(acc)),residence_NLL_nats=float(np.mean(wnll)),reference='Held-week empirical conditioned count model on4.13million moves: NLL0.64690, accuracy0.76374; different sample and access, not a controlled performance comparison',actual_query_GPS_used=False,Train_actual_prefix_used_for_teacher_diagnostic=True,Validation_or_Test_paths_decoded=False,Test_access=False)
write(OUT/'diagnostico_ajuste_dinamica_neural.json',report);print(json.dumps(report))

