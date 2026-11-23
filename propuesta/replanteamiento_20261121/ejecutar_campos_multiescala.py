"""Fresh architecture trial; execute matched capacity only after a real primary gain."""
from base_replanteo import *
import subprocess
proof=read(OUT/'verificacion_campo_multiescala.json');assert proof['passed']
sources=['entrenar_campo_multiescala.py','modelo_campo_multiescala.py','datos_campo_multiescala.py','base_replanteo.py']
hashes={f:sha(OUT/f) for f in sources}
plan=dict(hypothesis='Local endpoint patterns may be underrepresented by the existing global numerical embedding. Learn continuous multiscale origin/destination fields jointly from scratch.',
 architecture='Six31.25..1000m bilinear fields per endpoint,8channels each; concatenate origin,destination,elementwise product and8calendar signals;128SiLU projection into the original8component route-duration raw parameters. Global TabM and field both learned fresh.',
 regularization='Coordinate-only Train node support n/(n+20); unsupported fine locations back off to global/coarse representation.',
 information='Same five raw query fields,216derived legal features; actual path only original Train geometry loss. No new query metadata.',
 losses='Same primary NLL+5RPS+0.2conditional geometry likelihood. No failed semantic or privileged-teacher auxiliary.',
 budget='50epochs maximum, at least20 before patience stop,12checks without0.01s improvement,1hour per arm.512batch,FP32highest,AdamW.001.',
 control='Conditional parameter-matched dense branch1220x1220 if spatial model beats original full-cohort reference; identical initial global weights, ctor, optimizer, seed and shuffle/dropout streams checked.',
 evidence='CPU interpolation,zero-branch recovery,finite nonzero embedding/output gradients passed. Mandatory actual GPU gradient gate and separate exact protected-reference reproduction before updates.',
 context=[dict(title='ODConNet',url='https://openreview.net/pdf?id=y9tJg8Me65',status='Manuscript labeled under review forICLR2026; final acceptance not established here',relevance='Multiscale cells as a conceptual OD representation antecedent; not a local replication or Porto performance guarantee'),
 dict(title='MDTI',url='https://arxiv.org/html/2510.05840v1',status='arXiv2025 preprint',relevance='Uses full trajectories for route-based TTE; not a fair OD benchmark. Do not pass its observed GPS/grid sequence to query.')],
 rationale_after_failures='Request labels,privileged soft-target warm adjustment and frozen profile integration all failed full Validation. Change representation and fresh learning dynamics instead of additional frozen-output weights.',
 conservation='Retain every full epoch below131.06 and every strict new best below reference; shared original model and all TransferTraj comparisons protected.',
 sources=hashes,goal='active',Test='sealed')
write(OUT/'plan_campos_multiescala.json',plan)
results=[];kept=[]
for mode,runid in [('spatial','campo_multiescala_s0'),('capacity','control_capacidad_campo_s0')]:
 if mode=='capacity':
  first=results[0]
  if first['validation']['mae_s']>=131.25671207822356:
   write(OUT/'pipeline_campos_estado.json',dict(stage='Spatial field failed original reference; no extra capacity trial justified. Read full conclusion and continue distinct work.',goal='active',Test='sealed'));break
  if first['validation']['mae_s']<73.5:
   write(OUT/'pipeline_campos_estado.json',dict(stage='Target Validation candidate requires freeze/raw reproduction/confirmation. No extra training launched.',goal='active',Test='sealed'));break
 run=OUT/runid
 if not (run/'resultado.json').exists():
  assert not run.exists(),'Do not overwrite/restart incomplete trials; inspect actual terminal state first'
  for f,h in hashes.items():assert sha(OUT/f)==h
  with (OUT/(runid+'.log')).open('w',encoding='utf-8') as log:
   p=subprocess.Popen([sys.executable,'-B',str(OUT/'entrenar_campo_multiescala.py'),'--mode',mode],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
   active=dict(experiment=runid,pid=p.pid,supervisor_pid=os.getpid(),device='one GPU after CPU preprocessing',sources_frozen=True)
   c=read(OUT/'CONTINUIDAD_OPERATIVA.json');c['current_trial']=active;write(OUT/'CONTINUIDAD_OPERATIVA.json',c)
   write(OUT/'pipeline_campos_estado.json',dict(stage='Fresh field architecture active; inspect actual initial GPU gate and log',active=active,goal='active',Test='sealed'))
   while p.poll() is None:time.sleep(15)
  if not (run/'resultado.json').exists():
   write(OUT/'pipeline_campos_estado.json',dict(stage='Fresh stage ended without complete result; inspect log and checkpoint, no automatic overwrite',exit_code=p.returncode,goal='active',Test='sealed'));exit()
 r=read(run/'resultado.json');results.append(r)
 for item in r['saved_checkpoints']:
  assert sha(run/item['file'])==item['sha256'];kept.append(dict(id=runid,**item,reproduction_pending=True))
 write(OUT/'candidatos_conservados_campos.json',dict(checkpoints=kept,original_reference_protected=True,Test='sealed'))
 write(OUT/'conclusion_campos_multiescala.json',dict(results=results,retained_checkpoints=kept,goal='active',Test='sealed',next='Reproduce promising models; if none reach confirmation goal, draw critical conclusions and continue a distinct justified alternative. Individual trial limits never complete the goal.'))
if len(results)==2:
 a=dict(np.load(OUT/'campo_multiescala_s0/validacion.npz'));b=dict(np.load(OUT/'control_capacidad_campo_s0/validacion.npz'))
 for key in ['row','timestamp','true']:assert np.array_equal(a[key],b[key])
 conclusion=read(OUT/'conclusion_campos_multiescala.json')
 conclusion['best_selected_field_vs_capacity']=paired(a['true'],a['pred'],b['pred'],a['timestamp'])
 write(OUT/'conclusion_campos_multiescala.json',conclusion)
 write(OUT/'pipeline_campos_estado.json',dict(stage='Fresh field/capacity study complete; inspect full results and continue goal',goal='active',Test='sealed'))
print(json.dumps(dict(stage='field_stage_finished',complete_models=len(results),goal='active',Test='sealed')),flush=True)
