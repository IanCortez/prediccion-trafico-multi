from common import *
from features import *
from array import array
import csv
assert not (DATA/"trayectorias_train.npz").exists()
d=np.load(DATA/"train_x.npz");wanted=set(d["row"].tolist())
cut=json.loads((DATA/"manifiesto.json").read_text())["validation"]["start_timestamp"]
started=time.time();packed=array("f");offsets=[0];rows=[];timestamps=[]
csv.field_size_limit(2**31-1)
with (ROOT/"tesis/porto/porto/train.csv").open(encoding="utf-8-sig",newline="") as f:
    for i,r in enumerate(csv.DictReader(f)):
        if i not in wanted:continue
        p=np.asarray(json.loads(r["POLYLINE"]),dtype=np.float64);ts=int(r["TIMESTAMP"])
        if ts+(len(p)-1)*15>=cut:continue
        xy=np.column_stack([(p[:,0]-ORIGIN[0])*111.195*np.cos(np.deg2rad(ORIGIN[1])),(p[:,1]-ORIGIN[1])*111.195]).astype(np.float32)
        packed.frombytes(xy.tobytes());offsets.append(offsets[-1]+len(p));rows.append(i);timestamps.append(ts)
        if len(rows)%200000==0:print(dict(trips=len(rows),points=offsets[-1],seconds=time.time()-started),flush=True)
coords=np.frombuffer(packed,dtype=np.float32).reshape(-1,2)
np.savez_compressed(DATA/"trayectorias_train.npz",xy=coords,offsets=np.array(offsets,dtype=np.int64),row=np.array(rows,dtype=np.int32),timestamp=np.array(timestamps,dtype=np.int64))
write_json(DATA/"trayectorias_train_manifiesto.json",dict(n=len(rows),points=len(coords),xy_unit="km relative fixed geographic origin",sha256=sha(DATA/"trayectorias_train.npz"),train_manifest_sha256=sha(DATA/"manifiesto.json"),arrival_embargo=True,seconds=time.time()-started))
print(dict(trips=len(rows),points=len(coords),seconds=time.time()-started),flush=True)

