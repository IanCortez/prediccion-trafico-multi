"""Bounded CPU controlflow audit of releasedUNITE.forward; no trainedweights or Porto data."""
from base_replanteo import *
import ast,torch,random
start=time.time();torch.set_num_threads(2)
source=OUT/'literatura_UNITE_2026'
code=(source/'models_tte.py').read_text(encoding='utf-8-sig')
run=OUT/'auditoria_inferencia_UNITE_CPU_s0';assert not run.exists();run.mkdir()
def extract_forward(body):
 tree=ast.parse(body)
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='UNITE')
 fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
 module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
 namespace=dict(torch=torch,random=random)
 exec(compile(module,str(source/'models_tte.py'),'exec'),namespace)
 return namespace['forward']
official=extract_forward(code)
needle='            if stop.all():\n                break\n'
assert code.count(needle)==1
patched_code=code.replace(needle,'            # Controlprobe: continue to the fixedhorizon after individualarrival.\n')
patched=extract_forward(patched_code)
class Candidates:
 max_deg=1
 def __call__(self,input_id,dest,*args):
  B=len(input_id);next_id=torch.where(input_id<dest,input_id+1,dest)
  return torch.zeros(B,1,1),next_id[:,None],torch.ones(B,1),torch.zeros(B,1),torch.zeros(B,1)
class ControlledUNITE:
 seg_size=20;num_layers=1;num_slots=96;hid_dim=1;freq_limit=1
 candi_encoder=Candidates()
 def odt_encoder(self,o,d,*args):return torch.zeros(len(o),1)
 def emb_departure(self,t):return torch.zeros(len(t),1)
 def get_seg_emb(self,ids,*args):return ids.float()[:,None]
 def get_dest_info(self,d,slots):return d,None,None,None,None
 def prog_state_encoding(self,input_id,current,hidden,history,t):return current,hidden
 def eta_predictor(self,odt,prediction,input_id,*args):return torch.ones(len(input_id)),torch.zeros(len(input_id),1)
 def eta_fuse_mlp(self,x):return x[:,:1]
 def fc_id_out(self,x):return torch.zeros_like(x[...,:1])
def evaluate(forward,destinations,private_length=1,private_time=0.):
 B=len(destinations);zero=torch.zeros(B,dtype=torch.long)
 out=forward(ControlledUNITE(),zero,torch.tensor(destinations),zero,[private_length]*B,
  torch.zeros((private_length,B),dtype=torch.long),torch.full((private_length,B),private_time),zero,zero,-1)
 tte,prob,ids,length=out
 sums=[float(tte[:int(length[i]),i].sum()) for i in range(B)]
 return dict(predicted_segment_time_sums=sums,generated_lengths=length.tolist())
single=evaluate(official,[1]);mixed=evaluate(official,[1,3]);private=evaluate(official,[1],private_length=37,private_time=1234.)
single_fixed=evaluate(patched,[1]);mixed_fixed=evaluate(patched,[1,3])
assert single['generated_lengths']==[2] and mixed['generated_lengths']==[2,4]
assert single['predicted_segment_time_sums']==[1.]
assert mixed['predicted_segment_time_sums']==[2.,3.]
assert single['predicted_segment_time_sums']==private['predicted_segment_time_sums']
assert single_fixed['predicted_segment_time_sums']==[2.] and mixed_fixed['predicted_segment_time_sums']==[2.,4.]
assert single_fixed['predicted_segment_time_sums'][0]==mixed_fixed['predicted_segment_time_sums'][0]
proof=dict(complete=True,scope='Releasedforward controlflow under deterministicconstant1persegment ETA stubs; no trainedUNITE or PortoMAE.',
 official_singleton=single,official_mixed_batch=mixed,originalquery_result_changes_with_batch_partner=True,
 actual_query_length_and_times_unused_in_ODbranch_in_thisprobe=True,
 minimal_no_globalbreak_control= dict(singleton=single_fixed,mixed_batch=mixed_fixed),
 finding='Globalstop.all break can leave destinationETA at fixedlastslot49, outside the generated-length slice. A shortquery has its destinationETA included only when anotherquery keeps the loopalive.',
 inference_limits=[
 'This is a concretealgorithmic invariancefailure in the releasedcontrolflow, not a remeasurement of publishedaccuracy or a claim about paperintent.',
 'The fixed50ODhorizon and predictedactual_length mean thisaudit does NOT establish a truequeryN leakage.',
 'Sourcegen_speed explicitlyreads traj_train_100; no allegation that its speedprofile useshelddata.',
 'Removingglobalbreak is only an isolatingCPUcontrol, not a productionpatch. A Portoimplementation should finalize destinationETA perquery and use absorbingcompletedstates with correctboundedfallback.',
 'Any proposedfullUNITEadaptation must rerun raw5/batch/reverse/singleton invariance before a benchmarkcomparison.'
 ],
 sources={name:sha(source/name) for name in ['models_tte.py','prepare_workspace.py','infer_ODTTE.py','LICENSE']},
 URLs=['https://github.com/derekwtian/UNITE/blob/main/models/tte.py','https://github.com/derekwtian/UNITE/blob/main/infer_ODTTE.py','https://github.com/derekwtian/UNITE/blob/main/prepare_workspace.py'],
 source_license='MIT; exactLICENSE retained.',source_module_executed=False,only_extracted_forward_executed_with_controlled_stubs=True,
 GPU_used=False,Porto_data_access=False,Validation=False,Test=False,seconds=time.time()-start,own_verifier_sha256=sha(__file__))
write(run/'resultado.json',proof);(run/'verificador_ejecutado.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(complete=True,official_singleton=single,official_mixed_batch=mixed,control_fixed=single_fixed,no_Portodata_or_GPU=True)))
