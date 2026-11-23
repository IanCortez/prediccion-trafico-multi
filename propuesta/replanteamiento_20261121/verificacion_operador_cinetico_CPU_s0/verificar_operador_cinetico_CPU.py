"""Independentdense and exhaustivewalk oracle, fullfold intervention and rawqueryinvariants."""
from base_replanteo import *
from operador_red_cinetica import RoadClockKernel
from conteos_red_cinetica import encode_pairs,encode_events,aggregate_keys
import pandas as pd
start=time.time()
run=OUT/'verificacion_operador_cinetico_CPU_s0';assert not run.exists();run.mkdir()
K=24;src=np.array([0,0,1,1,1,2,2,3,3]);dst=np.array([0,1,0,1,2,2,3,3,1])
w0=np.array([2,5,1,2,5,2,6,2,1]);w1=np.array([6,3,1,4,4,7,3,3,2])
pair_keys=[];pair_weights=[];stop_keys=[];stop_weights=[];start_keys=[];start_weights=[]
for fold,multiple in [(1,1),(2,2)]:
 for hour,w in [(0,w0),(1,w1)]:
  pair_keys.extend(encode_pairs(np.full(len(src),fold*168+hour),src*3,dst*3,K));pair_weights.extend(w*multiple)
  stop_keys.extend(encode_events([fold*168+hour],[9],K));stop_weights.append((4 if hour==0 else 2)*multiple)
  start_keys.extend(encode_events([fold*168+hour,fold*168+hour],[0,3],K));start_weights.extend([10*multiple,3*multiple])
# Excludedfold has unique nodes; a correcttest must remove them BEFORE the catalogue/tree and everybackoff.
pair_keys.extend(encode_pairs([0,1],[15,18],[18,15],K));pair_weights.extend([9,5])
stop_keys.extend(encode_events([0],[18],K));stop_weights.append(4)
start_keys.extend(encode_events([0],[15],K));start_weights.append(7)
def unique(keys,weights):return aggregate_keys(np.array(keys,np.uint64),np.array(weights,np.uint64))
pk,pw=unique(pair_keys,pair_weights);sk,sw=unique(stop_keys,stop_weights);ik,iw=unique(start_keys,start_weights)
counts=dict(kinetic_pair_keys=pk,kinetic_pair_counts=pw,stop_keys=sk,stop_counts=sw,start_keys=ik,start_counts=iw,kinetic_state_count=np.array(K))
class Geometry:
 def centres(self,nodes,kind):
  spatial=nodes//(3 if kind=='kinetic' else 1)
  return np.column_stack([spatial*100.,np.zeros(len(nodes))]),np.full(len(nodes),100.)
geometry=Geometry();model=RoadClockKernel(counts,geometry,kind='position',excluded_fold=0)
assert np.array_equal(model.nodes,np.arange(4)) and model.size==4
# Manualdenseoracle, independentof encodedCSRconstruction.
Cs=[];stops=[];starts=[]
for w,stop in [(w0,4),(w1,2)]:
 C=np.zeros((4,4))
 for i,j,value in zip(src,dst,w):C[i,j]+=3*value
 Cs.append(C);s=np.zeros(4);s[3]=3*stop;stops.append(s);starts.append(np.array([30.,9.,0.,0.]))
globalC=Cs[0]+Cs[1];globalStop=stops[0]+stops[1];globalStart=starts[0]+starts[1]
total=globalC.sum(1)+globalStop;P=globalC/total[:,None];r=globalStop/total
gp=(globalStart+np.ones(4)/4)/(globalStart.sum()+1)
Q=[];R=[];A=[]
for C,stop,initial in zip(Cs,stops,starts):
 Nh=C.sum(1)+stop
 Q.append((C+10*P)/(Nh+10)[:,None]);R.append((stop+10*r)/(Nh+10))
 A.append((initial+10*gp)/(initial.sum()+10))
for hour in [0,1]:
 QT,stop,initial=model.kernel(hour)
 assert np.max(np.abs(QT.toarray().T-Q[hour]))<1e-15
 assert np.max(np.abs(stop-R[hour]))<1e-15 and np.max(np.abs(initial-A[hour]))<1e-15
timestamp=int(pd.Timestamp('2014-01-06 00:59:30',tz='UTC').timestamp())
sx=111195*np.cos(np.deg2rad(41.16))
query=dict(o_lon=np.array([-8.62+50/sx]),o_lat=np.array([41.16]),d_lon=np.array([-8.62+300/sx]),d_lat=np.array([41.16]),timestamp=np.array([timestamp]))
out=model.predict(query)
hours=np.array([0,0]+[1]*118)
ow=np.exp(-.5*np.array([50.,50.,150.,250.])**2/900);ow[2:]=0
initial=A[0]*ow;initial/=initial.sum()
LD=np.array([0.,0.,0.,1.]);flow=initial.copy();expected=np.zeros(116)
for n in range(120):
 if n>=4:expected[n-4]=flow@(R[hours[n]]*LD)
 if n<119:flow=flow@Q[hours[n]]
dense_error=float(np.max(np.abs(out['weights'][0]-expected)));assert dense_error<1e-13,dense_error
# Independentallwalk enumeration through8ticks; terminalobservation is permitted after an earlierdestinationvisit.
enumerated=np.zeros(9);path_count=0;return_visits=0
def walk(node,depth,mass,seen_dest):
 global path_count,return_visits
 h=hours[depth]
 if depth>=4:
  contribution=mass*R[h][node]*LD[node];enumerated[depth]+=contribution
  if contribution>0:
   path_count+=1
   if seen_dest:return_visits+=1
 if depth<8:
  for to,p in enumerate(Q[h][node]):
   if p>0:walk(to,depth+1,mass*p,seen_dest or node==3)
for node,mass in enumerate(initial):
 if mass>0:walk(node,0,mass,False)
enum_error=float(np.max(np.abs(enumerated[4:]-expected[:5])));assert enum_error<1e-13 and return_visits>0
# Corrupt everyexcludedfold table, adding newnodes aswell; probabilities and catalogues muststay identical.
modified={key:value.copy() for key,value in counts.items()}
for key in ['kinetic_pair_counts','stop_counts','start_counts']:
 corresponding={'kinetic_pair_counts':'kinetic_pair_keys','stop_counts':'stop_keys','start_counts':'start_keys'}[key]
 divisor=K*K if key=='kinetic_pair_counts' else K
 mask=modified[corresponding]//np.uint64(divisor)//168==0
 modified[key]=modified[key].astype(np.uint64);modified[key][mask]+=1234
new_key=encode_pairs([0],[21],[23],K)
modified['kinetic_pair_keys'],modified['kinetic_pair_counts']=unique(np.r_[modified['kinetic_pair_keys'],new_key],np.r_[modified['kinetic_pair_counts'],999])
for key,state in [('start',21),('stop',23)]:
 new_key=encode_events([0],[state],K)
 modified[key+'_keys'],modified[key+'_counts']=unique(np.r_[modified[key+'_keys'],new_key],np.r_[modified[key+'_counts'],777])
changed=RoadClockKernel(modified,geometry,kind='position',excluded_fold=0);after=changed.predict(query)
assert np.array_equal(model.nodes,changed.nodes) and np.array_equal(out['weights'],after['weights']) and np.array_equal(out['probability'],after['probability'])
kinetic=RoadClockKernel(counts,geometry,kind='kinetic',excluded_fold=0)
changedkin=RoadClockKernel(modified,geometry,kind='kinetic',excluded_fold=0)
assert np.array_equal(kinetic.predict(query)['weights'],changedkin.predict(query)['weights'])
assert np.array_equal(kinetic.predict(query)['weights'],out['weights']),'Singlemotionclass positioncollapse should be identical'
# Partner/reverse/singleton, out-of-support and no privatequeryfields.
batch={key:np.repeat(value,3) for key,value in query.items()};batch['d_lon'][2]=10.
batch_out=model.predict(batch);reverse=model.predict({key:value[::-1] for key,value in batch.items()})
assert np.array_equal(batch_out['weights'],reverse['weights'][::-1]) and np.array_equal(batch_out['weights'][:1],out['weights'])
assert not batch_out['supported'][2] and len(batch_out['supported'])==3
try:model.predict(dict(query,trueN=np.array([7])))
except AssertionError:private_fields_rejected=True
else:raise AssertionError('Extraqueryfield accepted')
for hour in range(8):model.kernel(hour)
assert model.cache_peak<=4 and len(model.cache)<=4
proof=dict(passed=True,scope='Independenttinygraph probabilities, clock andfullsourcefold exclusion; no PortoMAE.',
 dense_timevarying_weight_maxerror=dense_error,exhaustive_walk_weight_maxerror=enum_error,
 explicit_terminal_walks=path_count,repeated_destination_terminalwalks=return_visits,
 excludedfold_pair_start_stop_globalbackoff_and_catalogue_intervention_exact=True,
 kinetic_andposition_interventions_passed=True,singlemotionclass_collapse_exact=True,
 raw5_reverse_partner_singleton_exact=True,unsupported_rows_retained=True,privatequeryfields_rejected=private_fields_rejected,
 bounded4contextLRU=True,all_conditional_probability_sums_one=bool(np.allclose(out['probability'].sum(1),1)),
 no_true_queryN_or_path=True,GPU_used=False,Portodata_used=False,Validation=False,Test=False,seconds=time.time()-start,
 sources={name:sha(OUT/name) for name in ['operador_red_cinetica.py','verificar_operador_cinetico_CPU.py','conteos_red_cinetica.py','coordenadas_trayecto_2d.py']})
write(run/'resultado.json',proof)
for name in proof['sources']:(run/name).write_bytes((OUT/name).read_bytes())
print((run/'resultado.json').read_text(encoding='utf-8'),flush=True)
