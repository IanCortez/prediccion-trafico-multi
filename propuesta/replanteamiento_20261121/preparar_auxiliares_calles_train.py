"""Bounded resumable stream of Train-only partial street supervision. No primary filter changes."""
from base_replanteo import *
import csv,shutil,importlib.util
from numba import set_num_threads
from consulta_mapa2014 import resources,ORIGIN
from map_matching import project_candidates
import _trazador_viterbi_20261122 as trace_module
sys.modules['trazador_viterbi_20261122']=trace_module
access=deny_test();start=time.time();set_num_threads(2);run=OUT/'datos_calles_parciales_Train_s0';run.mkdir(exist_ok=True)
assert read(OUT/'oraculo_tiempo_calle_train_s0/resultado.json')['passed']
assert read(OUT/'verificacion_unite_neural_absorcion.json')['passed']
source=run/'fuente_ejecutada.py'
if source.exists():assert sha(source)==sha(__file__),'Resume requires immutable source'
else:shutil.copy2(__file__,source)
if (run/'resultado.json').exists():print('Already complete; no repeated preprocessing.');raise SystemExit(0)
memory_path=OUT/'datos_prototipos_historicos_s0/memoria_Train.npz'
m=dict(np.load(memory_path));assert len(m['row'])==894105
assert np.all(m['timestamp']+m['time_s']<1398345209)
plan=dict(scope='Train-only auxiliary street labels; all original primary queries/labels/splits unchanged.',trips=len(m['row']),
 matching='Frozen nearest4/Viterbi traced without output changes; no tolerance modifications.',
 labels='Original15seconds intervals distributed on public shortest bridges, merged into53766chains; uncertain intervalGAP65535. All totalduration labels unchanged.',
 quality='Existing894105quality eligibleTrain sources used only for auxiliary labels. All1023485purgedprimaryTrain queries retained in future objective.',
 privacy='BinaryCSVrow membership checked before decoding; no Validation/Test trajectories decoded.',
 endpoint='TrueHMM first/lastchain andfraction are auxiliary labels only; publicOD input must use standalone4projections from5fields.',
 resources=dict(chunk_trips=4096,max_artifact_MiB=500,CPU_threads=2,no_GPU=True),
 source_memory_sha256=sha(memory_path),source_sha256=sha(source),trace_sha256=sha(OUT/'_trazador_viterbi_20261122.py'),
 Test='sealed',resume='Completed immutable shards and recordedlastCSVrow are retained; no repeat of complete shards.')
assert plan['source_memory_sha256']=='0bc964fe70856e6cf5f62c926629ec95021afc96e7c3213eb493589f39b0da84'
if (run/'plan.json').exists():assert read(run/'plan.json')==plan
else:write(run/'plan.json',plan)
state_path=run/'progreso.json'
if state_path.exists():state=read(state_path)
else:state=dict(completed_trips=0,last_csv_row=-1,tokens=0,trusted_time_s=0.,total_time_s=0.,max_auxiliary_sum_error_s=0.,bridge_failures=0,overflow_trips=0,shards=[],active_seconds=0.)
for entry in state['shards']:
 file=run/entry['file'];assert sha(file)==entry['sha256']
raw,comp,sample_edges,tree,routes,profiles=resources();set_num_threads(2);D,conn,routing=routes['length']
flat=comp['flat_incoming_arc'];valid=flat>=0;pc=np.repeat(np.arange(len(comp['chain_src']),dtype=np.int32),np.diff(comp['chain_offsets']))
edge_chain=np.full(len(raw['u']),-1,np.int32);edge_chain[flat[valid]]=pc[valid]
edge_position=np.empty(len(raw['u']),np.int32);edge_position[flat[valid]]=np.flatnonzero(valid)
edge_prefix=comp['prefix_length'][edge_position-1];assert (edge_chain>=0).all() and (edge_prefix>=0).all()
helper_path=run/'helpers_parciales.py';prior_helper=OUT/'oraculo_tiempo_calle_train_s0/helpers_parciales.py'
if helper_path.exists():assert sha(helper_path)==sha(prior_helper)
else:shutil.copy2(prior_helper,helper_path)
spec=importlib.util.spec_from_file_location('helpers_calles_completas',helper_path);helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
lookup=np.full(int(m['row'].max())+1,-1,np.int32);lookup[m['row'].astype(np.int64)]=np.arange(len(m['row']),dtype=np.int32)
active_before=state['active_seconds']
def process(batch):
 indices=np.asarray([a for a,p,r in batch],np.int32);rows=np.asarray([r for a,p,r in batch],np.int64)
 ll=np.vstack([p for a,p,r in batch]);offset=np.r_[0,np.cumsum([len(p) for a,p,r in batch])].astype(np.int64)
 points=np.column_stack([(ll[:,0]-ORIGIN[0])*111195*np.cos(np.deg2rad(ORIGIN[1])),(ll[:,1]-ORIGIN[1])*111195])
 _,near=tree.query(points,k=32,workers=2);edges,frac,error=project_candidates(points,near,sample_edges,raw['xy'],raw['u'],raw['v'])
 be,bf,ber,resets,flags=trace_module.viterbi_trace(points,offset,edges,frac,error,raw['u'],raw['v'],raw['length'],D,*conn)
 codes=[];times=[];distances=[];masks=[];value_masks=[];ptr=[0];initial=[];end=[];sourcefold=[]
 local_trusted=0.;local_total=0.;local_max=0.;local_fails=0;local_overflow=0
 for i,(a,b) in zip(indices,zip(offset[:-1],offset[1:])):
  c,t,dist,unknown,fails,stationary,overflow=helper.one_trip(a,b,be,bf,flags,ber,raw['u'],raw['v'],raw['length'],D,conn,routing,edge_chain)
  expected=float(m['time_s'][i]);err=abs(float(t.astype(np.float64).sum())-expected);assert err<.001
  known=c!=65535;mask=np.zeros(len(c),bool);value_mask=np.zeros(len(c),bool)
  if len(c)>1:
   ix=np.flatnonzero(known[:-1]&known[1:]);mask[ix]=comp['chain_dst'][c[ix].astype(np.int32)]==comp['chain_src'][c[ix+1].astype(np.int32)]
   value_mask[1:]=mask[:-1]
  initial_offset=float(edge_prefix[be[a]]+bf[a]*raw['length'][be[a]])
  end_offset=float(edge_prefix[be[b-1]]+bf[b-1]*raw['length'][be[b-1]] )
  value_mask[0]=bool(known[0] and c[0]==edge_chain[be[a]] and ber[a]<=100)
  codes.append(c);times.append(t);distances.append(dist);masks.append(mask);value_masks.append(value_mask);ptr.append(ptr[-1]+len(c))
  initial.append(initial_offset);end.append(end_offset)
  local_trusted+=float(t[known].astype(np.float64).sum());local_total+=expected;local_max=max(local_max,err);local_fails+=fails;local_overflow+=overflow
 file=run/('parte_'+str(len(state['shards'])).zfill(4)+'.npz');assert not file.exists(),'Uncommitted shard: inspect explicitly before resume'
 np.savez_compressed(file,row=rows,memory_index=indices,train_index=m['train_index'][indices],timestamp=m['timestamp'][indices],fold=m['fold'][indices],
  true_duration_s=m['time_s'][indices],offsets=np.asarray(ptr,np.int64),chain=np.concatenate(codes),duration_s=np.concatenate(times),distance_m=np.concatenate(distances),
  next_route_valid=np.concatenate(masks),remaining_value_valid=np.concatenate(value_masks),
  initial_chain=edge_chain[be[offset[:-1]]].astype(np.uint16),initial_offset_m=np.asarray(initial,np.float32),
  terminal_chain=edge_chain[be[offset[1:]-1]].astype(np.uint16),terminal_offset_m=np.asarray(end,np.float32))
 entry=dict(file=file.name,trips=len(indices),tokens=ptr[-1],last_csv_row=int(rows[-1]),sha256=sha(file),bytes=file.stat().st_size)
 state['shards'].append(entry);state['completed_trips']+=len(indices);state['tokens']+=ptr[-1];state['last_csv_row']=int(rows[-1])
 state['trusted_time_s']+=local_trusted;state['total_time_s']+=local_total;state['max_auxiliary_sum_error_s']=max(state['max_auxiliary_sum_error_s'],local_max)
 state['bridge_failures']+=int(local_fails);state['overflow_trips']+=int(local_overflow);state['active_seconds']=active_before+time.time()-start
 assert sum(e['bytes'] for e in state['shards'])<plan['resources']['max_artifact_MiB']*1024**2
 write(state_path,state)
 print(json.dumps(dict(stage='Train_street_auxiliary_stream',completed=state['completed_trips'],total=len(m['row']),tokens=state['tokens'],MiB=sum(e['bytes'] for e in state['shards'])/1024**2,seconds=state['active_seconds'])),flush=True)
batch=[]
with (ROOT/'tesis/porto/porto/train.csv').open('rb') as f:
 header=next(csv.reader([next(f).decode('utf-8-sig').strip()]));pcol=header.index('POLYLINE');tcol=header.index('TIMESTAMP')
 for rawrow,line in enumerate(f):
  if rawrow<=state['last_csv_row'] or rawrow>=len(lookup):continue
  j=int(lookup[rawrow])
  if j<0:continue
  fields=next(csv.reader([line.decode('utf-8')]));assert int(fields[tcol])==m['timestamp'][j]
  p=np.asarray(json.loads(fields[pcol]),np.float64);assert (len(p)-1)*15==m['time_s'][j]
  batch.append((j,p,rawrow))
  if len(batch)>=4096:process(batch);batch=[]
 if batch:process(batch)
assert state['completed_trips']==len(m['row'])
result=dict(complete=True,scope=plan['scope'],trips=state['completed_trips'],tokens=state['tokens'],shards=state['shards'],
 trusted_time_fraction=state['trusted_time_s']/state['total_time_s'],max_auxiliary_sum_error_s=state['max_auxiliary_sum_error_s'],
 bridge_failures=state['bridge_failures'],overflow_trips=state['overflow_trips'],primary_cohort_unchanged=True,source_memory_sha256=plan['source_memory_sha256'],
 source_sha256=sha(source),helper_sha256=sha(helper_path),trace_sha256=plan['trace_sha256'],Validation_paths_decoded=False,Test_paths_decoded=False,
 artifact_MiB=sum(e['bytes'] for e in state['shards'])/1024**2,seconds=state['active_seconds'])
write(run/'resultado.json',result);print(json.dumps({k:result[k] for k in ['complete','trips','tokens','trusted_time_fraction','artifact_MiB','seconds']}),flush=True)
