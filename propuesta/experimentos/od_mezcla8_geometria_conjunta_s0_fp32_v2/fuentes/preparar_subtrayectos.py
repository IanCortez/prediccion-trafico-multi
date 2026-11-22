from common import *
import csv,math
tr=dict(np.load(DATA/"train_x.npz"));wanted=set(tr["row"].tolist())
cut=json.loads((DATA/"manifiesto.json").read_text())["validation"]["start_timestamp"]
assert not (DATA/"subtrayectos.npz").exists()
rng=np.random.default_rng(20261118);rows=[];rejected=0;started=time.time()
csv.field_size_limit(2**31-1)
with (ROOT/"tesis/porto/porto/train.csv").open(encoding="utf-8-sig",newline="") as f:
    for i,r in enumerate(csv.DictReader(f)):
        if i not in wanted:continue
        p=np.asarray(json.loads(r["POLYLINE"]),dtype=np.float64);ts=int(r["TIMESTAMP"]);n=len(p)
        if ts+(n-1)*15>=cut:continue
        for j in range(2):
            length=int(rng.integers(4,min(60,n-1)+1));start=int(rng.integers(0,n-length));end=start+length
            o,d=p[start],p[end]
            lat1,lat2=np.deg2rad([o[1],d[1]])
            a=np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(np.deg2rad(d[0]-o[0])/2)**2
            dist=6371000*2*np.arcsin(np.sqrt(np.clip(a,0,1)))
            if dist<200 or dist/(length*15)*3.6>120:rejected+=1;continue
            rows.append((i,ts+start*15,o[0],o[1],d[0],d[1],dist,length*15))
        if i%250000==0:print(dict(source_row=i,segments=len(rows),seconds=time.time()-started),flush=True)
a=np.array(rows,dtype=[("parent_row","i4"),("timestamp","i8"),("o_lon","f8"),("o_lat","f8"),("d_lon","f8"),("d_lat","f8"),("od_m","f4"),("duration","f4")])
np.savez_compressed(DATA/"subtrayectos.npz",**{k:a[k] for k in a.dtype.names})
write_json(DATA/"subtrayectos_manifiesto.json",dict(n=len(a),parents=int(len(np.unique(a["parent_row"]))),rejected=rejected,seed=20261118,train_manifest_sha256=sha(DATA/"manifiesto.json"),sha256=sha(DATA/"subtrayectos.npz"),start_max=int(a["timestamp"].max()),end_max=int((a["timestamp"]+a["duration"]).max()),validation_start=cut,label_information="Intermediate GPS and fixed 15s sampling from training trips only; no validation/test route data",method="Two random within-trip segments, 4..60 intervals, OD>=200m, OD speed<=120km/h, arrival before validation",seconds=time.time()-started))
print(dict(n=len(a),seconds=time.time()-started),flush=True)

