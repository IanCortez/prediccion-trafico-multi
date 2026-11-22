from common import *
from features import load_development
from historial import *
import pickle
started=time.time();tr,y,va,v=load_development();k=keys(tr);base=physical(tr)
fold=(tr["timestamp"]//(7*86400))%5
available=tr["timestamp"]+y<va["timestamp"].min()
out=None
for f in range(5):
    fit=np.flatnonzero((fold!=f)&available);q=np.flatnonzero(fold==f)
    enc=HistoricalEncoder().fit(k,y,base,fit)
    X,names=enc.transform({n:z[q] for n,z in k.items()},base[q])
    if out is None:out=np.zeros((len(y),X.shape[1]),dtype=np.float32)
    out[q]=X
    print(dict(fold=f,fit=len(fit),queries=len(q),seconds=time.time()-started),flush=True)
np.savez_compressed(DATA/"train_historial.npz",row=tr["row"],X=out,names=np.array(names),fold=fold)
enc=HistoricalEncoder().fit(k,y,base)
with (DATA/"historial_entrenado.pkl").open("wb") as f:pickle.dump(enc,f)
for split in ("validation","test"):
    d=dict(np.load(DATA/(split+"_x.npz")));X,names=enc.transform(keys(d),physical(d))
    np.savez_compressed(DATA/(split+"_historial.npz"),row=d["row"],X=X,names=np.array(names))
# Intervention test: changing query-fold labels cannot affect its encoding.
tiny=np.arange(min(2000,len(y)));kk={n:z[tiny] for n,z in k.items()}
fit=tiny[tiny%5!=0];q=tiny[tiny%5==0]
a=HistoricalEncoder().fit(kk,y[tiny],base[tiny],fit).transform({n:z[q] for n,z in kk.items()},base[q])[0]
changed=y[tiny].copy();changed[q]*=100
b=HistoricalEncoder().fit(kk,changed,base[tiny],fit).transform({n:z[q] for n,z in kk.items()},base[q])[0]
assert np.array_equal(a,b),"Query target leakage"
write_json(DATA/"historial_manifiesto.json",dict(n_features=len(names),labels_ending_after_validation_start_excluded=int((~available).sum()),fold_method="Departure UTC week index modulo 5; exclude complete query fold",shrinkage_specs=DEFS,normalizer="fixed 60 seconds plus OD at 20km/h, no fitted query label normalization",train_features_sha256=sha(DATA/"train_historial.npz"),encoder_sha256=sha(DATA/"historial_entrenado.pkl"),query_label_intervention_passed=True,validation_test_labels_used=False,seconds=time.time()-started))
print(dict(features=len(names),seconds=time.time()-started),flush=True)


