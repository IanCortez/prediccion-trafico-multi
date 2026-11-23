"""Execute unmodified official UNITE.forward on synthetic deterministic layer stubs."""
from base_replanteo import *
import ast,torch,torch.nn.functional as F,random,types,importlib.util
start=time.time();torch.set_num_threads(2)
repo=BASE/'referencias/UNITE';source=repo/'models/tte.py';tree=ast.parse(source.read_text(encoding='utf-8-sig'))
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='UNITE')
forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
layers=ast.parse((repo/'models/layers.py').read_text(encoding='utf-8-sig'))
mask=next(n for n in layers.body if isinstance(n,ast.FunctionDef) and n.name=='sequence_mask')
env=dict(torch=torch,F=F,random=random,math=__import__('math'),np=np)
exec(compile(ast.Module(body=[mask,forward],type_ignores=[]),str(source),'exec'),env)
class Candidates:
 max_deg=2
 def __call__(self,input_id,*args):
  nbr=torch.stack([(input_id+1).clamp_max(6),(input_id+2).clamp_max(6)],-1)
  emb=nbr.float()[...,None].expand(-1,-1,4)
  return emb,nbr,torch.ones_like(nbr),None,None
class Fixture:
 seg_size=6;hid_dim=4;num_layers=1;num_slots=24;freq_limit=1
 def __init__(self):
  self.candi_encoder=Candidates();self.eta_fuse_mlp=torch.nn.Identity()
 def odt_encoder(self,o,d,depart,discrete):return torch.stack([o.float(),d.float(),depart.float(),torch.ones_like(o).float()],-1)*.01
 def emb_departure(self,depart):return torch.zeros(len(depart),4)
 def get_seg_emb(self,input_id,*args):return input_id.float()[:,None].expand(-1,4)*.01
 def get_dest_info(self,*args):return None,None,None,None,None
 def prog_state_encoding(self,input_id,curr,hidden,seqs,t):
  value=curr+hidden[0]*.01;return value,value[None]
 def eta_predictor(self,odt,prediction,input_id,*args):return input_id.float()+1,prediction
 def fc_id_out(self,inputs):return -inputs[...,:1]
Fixture.forward=env['forward']
def run(length,ratio,poison=False,origins=None,destinations=None):
 o=torch.tensor([0,0] if origins is None else origins);d=torch.tensor([4,5] if destinations is None else destinations);B=len(o)
 seq=torch.full((length,B),123456 if poison and ratio==-1 else 0,dtype=torch.long)
 if ratio==1:seq=torch.arange(length)[:,None].expand(-1,B)%5
 times=torch.full((length,B),999999. if poison else 0.)
 random.seed(20261122)
 outputs=Fixture().forward(o,d,torch.zeros(B,dtype=torch.long),[length]*B,seq,times,torch.zeros(B,dtype=torch.long),torch.zeros(B,6),ratio)
 tte,prob,ids,actual=outputs;total=torch.stack([tte[:int(actual[i]),i].sum() for i in range(B)])
 return total,ids,actual
base=run(3,-1);poison=run(80,-1,True)
assert all(torch.equal(a,b) for a,b in zip(base,poison))
zero_short=run(3,0);zero_long=run(80,0,True);zero_length_sensitive=not torch.equal(zero_short[0],zero_long[0])
single=run(3,-1,origins=[0],destinations=[4]);mixed=run(3,-1,origins=[0,0],destinations=[4,5])
batch_sensitive=not torch.equal(single[0][:1],mixed[0][:1])
result=dict(scope='Official forward control-flow execution with deterministic synthetic layer stubs, not actual trained network or a Porto MAE experiment.',
 official_source_sha256=sha(source),layers_source_sha256=sha(repo/'models/layers.py'),
 extracted_method_unmodified=True,real_datasets_opened=False,Test_access=False,
 ratio_minus1_route_length_times_poisoning_invariant=True,
 ratio0_observed_length_sensitive_negative_control=zero_length_sensitive,
 synthetic_total_base=base[0].tolist(),synthetic_generated_lengths=base[2].tolist(),
 batch_composition_sensitive_with_positive_destination_stub=batch_sensitive,
 same_first_query_alone_synthetic_total=single[0][0].item(),same_query_in_mixed_batch_synthetic_total=mixed[0][0].item(),
 interpretation='Mode−1 removes dependence on true targetroute/length. However stopped rows can keep being updated until allrows stop; positive terminal-segment predictions expose a batch-composition hazard. Inspect real SegTimeEstimator for hard-zero terminal enforcement before attributing this to trained UNITE.',
 requirements=['Never execute ODdriver defaults: tf_ratio defaults1, which teacher-forces actual route.',
 'ODadapter must supply fixed dummytarget tensors and ratio−1, project endpoints from raw5/publicmap instead of true mapmatchedroute.',
 'Check/repair absorption and terminaltime placement, and require batch/singleton invariance for actual neural adapter.',
 'Keep original primary labels and every query; mapmatched segment labels Train auxiliary only.'],
 source_sha256=sha(__file__),seconds=time.time()-start,
 installed_packages={name:importlib.util.find_spec(name) is not None for name in ['torch_geometric','networkx','geopandas']})
write(OUT/'instrumento_unite_od_sintetico.json',result);print(json.dumps(result),flush=True)
