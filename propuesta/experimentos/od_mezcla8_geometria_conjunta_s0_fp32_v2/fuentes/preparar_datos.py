from common import *
import csv, math
from collections import Counter
SOURCE=ROOT/"tesis/porto/porto/train.csv"
EXPECTED="2891d03ea99ce0fc203daedd434848a1a8baa7a8fb6da10f8d7ff26559efff6d"
assert sha(SOURCE)==EXPECTED,"Datos originales alterados"
assert not (DATA/"manifiesto.json").exists(),"El conjunto ya está congelado"
exposed=set()
for name in ("central_6000.csv","general_5000.csv"):
    with (ROOT/"auditoria/muestras/porto"/name).open(encoding="utf-8-sig",newline="") as f:
        exposed.update(r["TRIP_ID"] for r in csv.DictReader(f))
csv.field_size_limit(2**31-1)
rows=[];count=Counter();started=time.time()
def inside(p):return -8.6887<=p[0]<=-8.5557 and 41.1405<=p[1]<=41.1865
with SOURCE.open(encoding="utf-8-sig",newline="") as f:
    for i,r in enumerate(csv.DictReader(f)):
        count["raw"]+=1
        if r["MISSING_DATA"].lower()=="true":count["missing"]+=1;continue
        p=json.loads(r["POLYLINE"]);n=len(p)
        if not 5<=n<=120:count["points"]+=1;continue
        if not inside(p[0]) or not inside(p[-1]):count["endpoints"]+=1;continue
        o,d=p[0],p[-1]
        lat1,lat2=map(math.radians,(o[1],d[1]))
        a=math.sin((lat2-lat1)/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(math.radians(d[0]-o[0])/2)**2
        dist=6371000*2*math.asin(min(1,math.sqrt(a)))
        dur=(n-1)*15
        if dist<200:count["od_short"]+=1;continue
        if dist/dur*3.6>120:count["od_fast"]+=1;continue
        rows.append((i,int(r["TIMESTAMP"]),o[0],o[1],d[0],d[1],dist,dur,r["CALL_TYPE"],r["TRIP_ID"] in exposed,all(inside(v) for v in p)))
        if i%250000==0:print(dict(rows=i,accepted=len(rows),seconds=time.time()-started),flush=True)
dtype=[("row","i4"),("timestamp","i8"),("o_lon","f8"),("o_lat","f8"),("d_lon","f8"),("d_lat","f8"),("od_m","f4"),("duration","f4"),("call","U1"),("exposed","?"),("all_inside","?")]
a=np.array(rows,dtype=dtype);del rows
a=a[np.lexsort((a["row"],a["timestamp"]))]
n=len(a);b1=int(n*.8);b2=int(n*.9)
b1=int(np.searchsorted(a["timestamp"],a["timestamp"][b1],side="left"))
b2=int(np.searchsorted(a["timestamp"],a["timestamp"][b2],side="left"))
splits={"train":np.arange(b1),"validation":np.arange(b1,b2),"test":np.arange(b2,n)}
manifest=dict(version=1,source=str(SOURCE),source_sha256=EXPECTED,counts=dict(count),n=n,bbox=[41.1405,41.1865,-8.6887,-8.5557],protocol_sha256=sha(BASE/"PROTOCOLO.md"),columns_inference=["o_lon","o_lat","d_lon","d_lat","timestamp"],secondary_metadata=["call"],test_accesses=0)
for name,ix in splits.items():
    # No observed route, point count or travel time enters x.
    np.savez_compressed(DATA/(name+"_x.npz"),**{k:a[k][ix] for k in ("row","timestamp","o_lon","o_lat","d_lon","d_lat","od_m","call","exposed")})
    np.save(DATA/(name+"_y.npy"),a["duration"][ix])
    manifest[name]=dict(n=len(ix),start_timestamp=int(a["timestamp"][ix[0]]),end_timestamp=int(a["timestamp"][ix[-1]]),x_sha256=sha(DATA/(name+"_x.npz")),y_sha256=sha(DATA/(name+"_y.npy")),previously_exposed=int(a["exposed"][ix].sum()),all_inside=int(a["all_inside"][ix].sum()))
manifest["confirmation_n"]=int((~a["exposed"][splits["test"]]).sum())
manifest["elapsed_seconds"]=time.time()-started
write_json(DATA/"manifiesto.json",manifest)
print(json.dumps(manifest,indent=2),flush=True)

