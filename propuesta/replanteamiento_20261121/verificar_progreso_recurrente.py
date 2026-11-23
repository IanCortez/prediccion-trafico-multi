"""Actual neural absorption and real full-data CUDA resource preflight; Train only."""
from base_replanteo import *
import torch,inspect,gc
from modelo_progreso_recurrente import ProgressNet
from datos_progreso_recurrente import CorrectedSequences,load_graph
from inferencia import feature_input
from modelo_aux_subviajes import duration_score
access=deny_test();start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
torch.cuda.set_per_process_memory_fraction(.5);torch.manual_seed(20261122)
g=dict(static=np.zeros((3,12),np.float32),chain_length=np.array([10,20,30],np.float32),
 chain_src=np.arange(3),chain_dst=np.arange(1,4),
 neighbors=np.array([[1,3,3,3,3,3],[2,3,3,3,3,3],[3,3,3,3,3,3]]),
 neighbor_valid=np.array([[1,0,0,0,0,0],[1,0,0,0,0,0],[0,0,0,0,0,0]],bool),
 D=np.array([[0,10,30,60],[np.inf,0,20,50],[np.inf,np.inf,0,30],[np.inf,np.inf,np.inf,0]],np.float32),
 summer_start=np.array([1396141200]),summer_end=np.array([1414285200]))
m=ProgressNet(np.zeros(216),np.ones(216),g).eval()
with torch.no_grad():
 m.time_head[-1].weight.zero_();m.time_head[-1].bias.zero_()
 m.policy[-1].weight.zero_();m.policy[-1].bias.zero_();m.eos.weight.zero_();m.eos.bias.fill_(100)
chain=torch.zeros((3,2,4),dtype=torch.long);chain[1,1]=2
fraction=torch.zeros((3,2,4));fraction[:,1]=.5
args=dict(x=torch.zeros(3,216),timestamp=torch.tensor([1400000000,1400000001,1400000002]),
 chain=chain,fraction=fraction,error_m=torch.zeros(3,2,4),remaining_km=torch.ones(3,4,4),
 reachable=torch.ones(3,4,4,dtype=torch.bool))
args['reachable'][2]=False
hidden=[]
hook=m.gru.register_forward_pre_hook(lambda module,inp:hidden.append(inp[1].detach().clone()))
out=m.public_rollout(**args,max_steps=8,top_pairs=1);hook.remove()
rev=m.public_rollout(**{k:v.flip(0) for k,v in args.items()},max_steps=8,top_pairs=1).flip(0)
solo=torch.cat([m.public_rollout(**{k:v[i:i+1] for k,v in args.items()},max_steps=8,top_pairs=1) for i in range(3)])
assert torch.allclose(out,rev,atol=2e-4,rtol=1e-6) and torch.allclose(out,solo,atol=2e-4,rtol=1e-6)
assert len(hidden)==3 and torch.equal(hidden[1][:,:4],hidden[2][:,:4])
pace=.025+np.log(2)/5;delay=np.log(2)*5
expected=np.array([pace*5+delay,pace*45+3*delay])
# Styles are latent but localtimes fixed in this controlled actual-neural test.
prob=1/16
actual=(out[:2,17].numpy()-(1-prob)*out[:2,1].numpy())/prob
assert np.max(abs(actual-expected))<.002,(actual,expected)
assert abs(float(out[2,3])-float(out[2,1]))<.001 and float(out[2,5])==1
assert out[0,6]<out[1,6] and out[0,8]>0 and out[1,8]>0
# Float64 clock sees single seconds in 2014, without 128-second Float32 quantization.
cal=m.calendar(torch.tensor([1400000000.,1400000001.],dtype=torch.float64))
assert not torch.equal(cal[0],cal[1])
sig=inspect.signature(m.public_rollout)
assert not any(k in sig.parameters for k in ['route','route_length','target','duration','driver','teacher_forcing_ratio'])
try:m.public_rollout(**args,route=torch.ones(3))
except TypeError:pass
else:raise AssertionError('Private route accepted')
assert 'D' not in m.state_dict() and 'static' not in m.state_dict()
del m;gc.collect()

teacher=CorrectedSequences();d,X,names=feature_input('train',read(BEST/'resultado.json'))
eligible=teacher.eligible;mean=X[eligible].mean(0,dtype=np.float64).astype(np.float32)
std=X[eligible].std(0,dtype=np.float64).clip(.01).astype(np.float32)
g=load_graph();model=ProgressNet(mean,std,g).cuda();del g
x=torch.tensor(X,device='cuda');del X
bankdata=dict(np.load(OUT/'banco_valor_vial_s0/train_proyecciones.npz'))
bank={k:torch.as_tensor(bankdata[k].astype(np.int64) if k=='chain' else bankdata[k],device='cuda') for k in ['chain','fraction','error_m','remaining_km','reachable']}
del bankdata
rng=np.random.default_rng(20261122);fi=rng.choice(len(teacher.fragments),128,replace=False)
b=teacher.batch(fi);ti=b['train_index'];noise=torch.zeros(len(fi),4,device='cuda')
opt=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.0003)
torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.time()
records=[]
for it in range(8):
 model.train();opt.zero_grad(set_to_none=True)
 loss,details=model.teacher_forward(x[ti],b,noise)
 root=model.root(x[ti]);primary=duration_score(model.root_logprob(root),torch.as_tensor(teacher.y,device='cuda')[ti],model.grid)
 table=model.roads(torch.arange(model.nroads+1,device='cuda'))
 attach=model.attachment_logprob(root,**{k:v[ti] for k,v in bank.items()},road_table=table)
 av=b['attachment']>=0
 ce=torch.nn.functional.nll_loss(attach[av],b['attachment'][av])
 total=loss+primary+.1*ce;assert torch.isfinite(total)
 total.backward()
 assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 if it==0:
  for layer in [model.gru,model.embedding,model.time_head,model.prior,model.recognition]:
   assert sum(float(p.grad.abs().sum()) for p in layer.parameters() if p.grad is not None)>0
 torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
 records.append(float(total.detach()))
torch.cuda.synchronize();train_seconds=(time.time()-tick)/8
model.eval();ii=eligible[:64];ii=torch.tensor(ii,device='cuda');ts=torch.as_tensor(d['timestamp'],device='cuda')
torch.cuda.synchronize();tick=time.time()
summary=model.public_rollout(x[ii],ts[ii],**{k:v[ii] for k,v in bank.items()})
torch.cuda.synchronize();roll_seconds=time.time()-tick
rev=model.public_rollout(x[ii.flip(0)],ts[ii.flip(0)],**{k:v[ii.flip(0)] for k,v in bank.items()}).flip(0)
assert torch.allclose(summary,rev,atol=.01,rtol=2e-5)
statebytes=sum(v.numel()*v.element_size() for v in model.state_dict().values());assert statebytes<50*1024**2
result=dict(passed=True,model_source_sha256=sha(OUT/'modelo_progreso_recurrente.py'),data_source_sha256=sha(OUT/'datos_progreso_recurrente.py'),
 verifier_source_sha256=sha(__file__),actual_neural_terminal_time_positive_and_included_once=True,completed_hidden_frozen=True,
 controlled_short_long_mixed_reverse_singleton_consistent=True,Float64_clock_distinguishes_one_second=True,
 no_private_target_parameters=True,all_teacher_gradient_paths_finite_nonzero=True,
 full_Train_fragments=len(teacher.fragments),full_Train_tokens=len(teacher.code),resident_teacher_MiB=teacher.bytes/1024**2,
 B128_auxiliary_plus_primary_plus_attachment_step_seconds=train_seconds,
 public_64queries_16paths_max192_seconds=roll_seconds,
 public_allTrain_rollout_projected_minutes=roll_seconds*len(eligible)/64/60,
 GPU_peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,GPU_peak_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2,
 state_dict_MiB=statebytes/1024**2,D_persistent=False,losses=records,no_Validation_labels_or_Test_access=True,
 actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(OUT/'verificacion_progreso_recurrente.json',result)
print(json.dumps(result),flush=True)
