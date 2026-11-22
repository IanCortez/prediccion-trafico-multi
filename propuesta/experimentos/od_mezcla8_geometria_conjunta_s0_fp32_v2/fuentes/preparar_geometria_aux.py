from common import *
from features import *
import csv
started=time.time();q=np.linspace(0,1,9)[1:-1]
def summarize(xy):
    ds=np.sqrt(np.sum(np.diff(xy,axis=0)**2,axis=1))*1000
    s=np.r_[0,np.cumsum(ds)];total=s[-1]
    valid=np.r_[True,np.diff(s)>1e-7]
    s=s[valid];xx=xy[valid]
    pts=np.column_stack([np.interp(q*total,s,xx[:,i]) for i in (0,1)])
    straight=xy[0]*(1-q[:,None])+xy[-1]*q[:,None]
    return np.r_[total,(pts-straight).ravel(),ds.max()].astype(np.float32)
z=np.load(DATA/"trayectorias_train.npz");xy=z["xy"];offsets=z["offsets"];rows=z["row"]
arr=np.empty((len(rows),16),dtype=np.float32)
for i in range(len(rows)):
    arr[i]=summarize(xy[offsets[i]:offsets[i+1]])
    if i%250000==0:print(dict(train=i,seconds=time.time()-started),flush=True)
np.savez_compressed(DATA/"geometria_train.npz",row=rows,X=arr,quantiles=q)
d=dict(np.load(DATA/"validation_x.npz"));loc={int(v):i for i,v in enumerate(d["row"])}
out=np.empty((len(loc),16),dtype=np.float32)
csv.field_size_limit(2**31-1)
with (ROOT/"tesis/porto/porto/train.csv").open(encoding="utf-8-sig",newline="") as f:
    for i,r in enumerate(csv.DictReader(f)):
        if i not in loc:continue
        p=np.asarray(json.loads(r["POLYLINE"]),dtype=np.float64)
        points=np.column_stack([(p[:,0]-ORIGIN[0])*111.195*np.cos(np.deg2rad(ORIGIN[1])),(p[:,1]-ORIGIN[1])*111.195])
        out[loc[i]]=summarize(points)
np.savez_compressed(DATA/"geometria_validation.npz",row=d["row"],X=out,quantiles=q)
write_json(DATA/"geometria_manifiesto.json",dict(train_n=len(rows),validation_n=len(out),features="actual path length (m), 7 normalized-arclength deviations from OD chord (km), max GPS step (m); no number of points",usage="Training auxiliary targets and validation oracle diagnostics only. Never OD inference inputs.",test_path_read=False,train_sha256=sha(DATA/"geometria_train.npz"),validation_sha256=sha(DATA/"geometria_validation.npz"),seconds=time.time()-started))
print(dict(train=len(rows),validation=len(out),seconds=time.time()-started),flush=True)

