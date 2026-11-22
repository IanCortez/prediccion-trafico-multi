"""Complete prior lookup construction without rebuilding or altering saved Train counts."""
from base_replanteo import *
from historia_eventos import TemporalEventHistory
import torch,shutil
access=deny_test()
start=time.time()
assert read(OUT/'verificacion_historia_eventos.json')['passed']
folder=OUT/'prior_eventos'
assert (folder/'conteos_fold.npz').exists() and not (folder/'resultado.json').exists()
count_sha=sha(folder/'conteos_fold.npz')
hist=np.load(folder/'conteos_fold.npz')['counts']
diag=read(OUT/'diagnostico_riesgos_temporales.json')
assert hist.shape==(5,2376*120,120,2) and hist.dtype==np.uint16
assert int(hist.sum(dtype=np.uint64))==diag['kept_runs']
history=TemporalEventHistory(hist)
full=history.complete_probability()
assert np.isfinite(full).all() and np.allclose(full.sum(-1),1,atol=2e-7)
assert np.all(full[np.arange(len(full))//12%2==0,:,2]==0)
np.savez_compressed(folder/'probabilidad_train.npz',probability=full)
# Independently match all9,198,270 held point decisions of the earlier instrument.
total_loss=0.;points=0
for first in range(0,len(full),4096):
    last=min(first+4096,len(full))
    held=hist[0,first:last].astype(np.float64)
    events=held.sum(-1)
    risk=np.cumsum(events[:,::-1],axis=1)[:,::-1]
    actual=np.stack([risk-events,held[:,:,0],held[:,:,1]],-1)
    p=history.lookup(np.arange(first,last)[:,None],np.arange(120)[None,:],0)
    total_loss-=float((actual*np.log(np.maximum(p,1e-30))).sum())
    points+=int(risk.sum())
held_nll=total_loss/points
expected=diag['scores']['cell_heading_destination_clock_age']['held_NLL_per_point']
assert points==diag['scores']['cell_heading_destination_clock_age']['held_points']
assert abs(held_nll-expected)<1e-6,(held_nll,expected)
assert sha(folder/'conteos_fold.npz')==count_sha
assert not torch.cuda.is_initialized()
(folder/'fuentes').mkdir(exist_ok=True)
for name in ['preparar_historia_eventos.py','recuperar_historia_eventos.py','historia_eventos.py','verificar_historia_eventos.py']:shutil.copy2(OUT/name,folder/'fuentes'/name)
write(folder/'fallo_preparacion.json',dict(original_stage='Counts complete; lookup constructor failed its ndim assertion',original_history_code_sha256=sha(folder/'fallo_fuentes/historia_eventos.py'),original_log_sha256=sha(OUT/'preparar_historia_eventos.log'),recovery='Correct ndim4 and broadcasting; verify whole-fold intervention first. Reuse original count file unchanged, no repeated data counting.',counts_before_and_after_sha256=count_sha,recovered=True))
record=dict(Train_runs=diag['kept_runs'],histogram_shape=list(hist.shape),dtype=str(hist.dtype),maximum_count=int(hist.max()),full_probability_shape=list(full.shape),scope='Context-level Train event histograms, not query-specific prediction or hidden-state banks',key='(((cell*5+incoming_heading)*2+known_destination_equality)*12+run_entry_UTC4h_weekend)',event_age='Move atD-1; EOS atD; preceding points stay',smoothing=[1,50,20],Train_history='Exclude current entire departure-week fold from every backoff',Validation_history='Full eligible Train counts only',held_instrument_reproduced=dict(points=points,NLL=held_nll,previous_NLL=expected),no_GPU=True,Test_access=False,outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [folder/'conteos_fold.npz',folder/'probabilidad_train.npz']},code_sha256=sha(__file__),history_code_sha256=sha(OUT/'historia_eventos.py'),seconds=time.time()-start)
write(folder/'resultado.json',record)
print(json.dumps(dict(outputs=record['outputs'],held_instrument_reproduced=record['held_instrument_reproduced'],seconds=record['seconds'])),flush=True)
