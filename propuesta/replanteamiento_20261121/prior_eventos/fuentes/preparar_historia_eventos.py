"""Build only Train context-level histories, not per-query prediction banks."""
from base_replanteo import *
from historia_eventos import TemporalEventHistory
import torch,shutil
torch.set_num_threads(2)
access=deny_test()
start=time.time()
diag=read(OUT/'diagnostico_riesgos_temporales.json')
assert diag['passed'] and diag['worth_temporal_hazard_prior_test']
folder=OUT/'prior_eventos'
assert not folder.exists()
folder.mkdir()
tr=dict(np.load(DATA/'train_x.npz'))
data=np.load(OUT/'datos_semi_markov/runs_train.npz')
g=read(OUT/'datos_semi_markov/protocolo.json')
N=g['N'];H=120;groups=N*5*2*12
cell=data['cell'].astype(np.int32)
duration=data['duration_n'];enter=data['enter_n']
offset=data['run_offsets'];trip=data['trip_train_index']
valid=np.unpackbits(data['outgoing_valid_bits'])[:len(cell)].astype(bool)
xx=(tr['d_lon']+8.62)*111195*np.cos(np.deg2rad(41.16))
yy=(tr['d_lat']-41.16)*111195
dest=np.floor((xx-g['x0'])/200).astype(np.int32)*g['ny']+np.floor((yy-g['y0'])/200).astype(np.int32)
hist=np.zeros((5,groups,H,2),np.uint32)
kept=0
for first in range(0,len(cell),500000):
    rr=np.arange(first,min(first+500000,len(cell)))
    ti=np.searchsorted(offset,rr,side='right')-1
    ii=trip[ti]
    cur=cell[rr];previous=cell[np.maximum(rr-1,0)]
    terminal=rr==offset[ti+1]-1
    known=(rr>offset[ti])&(previous<N)&valid[np.maximum(rr-1,0)]
    dx=cur//g['ny']-previous//g['ny'];dy=cur%g['ny']-previous%g['ny']
    heading=np.where(known,np.where(np.abs(dx)>=np.abs(dy),np.where(dx>=0,0,1),np.where(dy>=0,2,3)),4)
    at_destination=cur==dest[ii]
    age=duration[rr].astype(np.int16)-(~terminal)
    good=(cur<N)&(age>=0)&(age<H)
    assert np.all((~terminal)|at_destination)
    timestamp=tr['timestamp'][ii]
    entered=timestamp+enter[rr].astype(np.int64)*15
    weekday=(entered//86400+3)%7
    clock=(entered%86400)//14400+6*(weekday>=5)
    group=((cur*5+heading)*2+at_destination)*12+clock
    fold=(timestamp//(7*86400))%5
    np.add.at(hist,(fold[good],group[good],age[good],terminal[good].astype(np.int64)),1)
    kept+=int(good.sum())
    if first%4000000==0:print(json.dumps(dict(stage='Train_event_histograms',completed_runs=first,total_runs=len(cell))),flush=True)
maximum=int(hist.max())
assert maximum<65536 and kept==diag['kept_runs']
hist=hist.astype(np.uint16)
np.savez_compressed(folder/'conteos_fold.npz',counts=hist)
del tr,data,cell,duration,enter,valid,xx,yy,dest
history=TemporalEventHistory(hist)
full=history.complete_probability()
assert np.isfinite(full).all() and np.allclose(full.sum(-1),1,atol=2e-7)
assert np.all(full[np.arange(groups)//12%2==0,:,2]==0)
np.savez_compressed(folder/'probabilidad_train.npz',probability=full)
held_group=np.arange(0,groups,997)
age=np.arange(len(held_group))%120
p=history.lookup(held_group,age,np.zeros(len(held_group),np.int64))
assert np.isfinite(p).all() and np.allclose(p.sum(-1),1,atol=2e-7)
assert not torch.cuda.is_initialized()
(folder/'fuentes').mkdir()
for name in ['preparar_historia_eventos.py','historia_eventos.py']:shutil.copy2(OUT/name,folder/'fuentes'/name)
record=dict(Train_runs=kept,histogram_shape=list(hist.shape),dtype=str(hist.dtype),maximum_count=maximum,full_probability_shape=list(full.shape),scope='Context-level event histograms from Train; not query-specific predictions or hidden states',key='(((cell*5+incoming_heading)*2+known_destination_equality)*12+run_entry_UTC4h_weekend)',event_age='Nonterminal move atD-1; terminal EOS atD; all earlier decisions stay',backoff='Global destination flag age, cell heading destination age, then UTC clock',smoothing=[1,50,20],Train_history='Exclude current entire departure-week fold at every backoff level',Validation_history='All eligible Train counts; no Validation GPS or labels',Test_access=False,no_GPU=True,actual_files_accessed=sorted(access),code_sha256=sha(__file__),history_code_sha256=sha(OUT/'historia_eventos.py'),outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [folder/'conteos_fold.npz',folder/'probabilidad_train.npz']},seconds=time.time()-start)
write(folder/'resultado.json',record)
print(json.dumps(dict(stage='complete',outputs=record['outputs'],seconds=record['seconds'])),flush=True)
