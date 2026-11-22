from common import *
from features import *
tr,y,va,v=load_development()
m=json.loads((DATA/"manifiesto.json").read_text(encoding="utf-8"))
checks={}
checks["temporal"]=int(tr["timestamp"].max())<int(va["timestamp"].min()) and int(va["timestamp"].max())<m["test"]["start_timestamp"]
checks["row_disjoint"]=not np.intersect1d(tr["row"],va["row"]).size
checks["permitted_columns"]=set(tr)==set(["row","timestamp","o_lon","o_lat","d_lon","d_lat","od_m","call","exposed"])
small={k:z[:100].copy() for k,z in va.items()}
X,names=features(small);small["duration"]=np.full(100,1e8);small["n_points"]=np.full(100,999)
small["route_len"]=np.arange(100)*1e7
X2,names2=features(small)
checks["label_route_invariance"]=bool(np.array_equal(X,X2) and names==names2)
alt={k:z.copy() for k,z in small.items()};alt["call"][:]="Z"
checks["strict_od_call_invariance"]=bool(np.array_equal(X,features(alt)[0]))
rev={k:z[::-1] for k,z in small.items()}
checks["permutation_equivariance"]=bool(np.array_equal(X[::-1],features(rev)[0]))
checks["duration_units"]=bool(np.all(y%15==0) and y.min()>=60 and y.max()<=1785)
checks["manifest_train_hash"]=sha(DATA/"train_y.npy")==m["train"]["y_sha256"]
checks["manifest_validation_hash"]=sha(DATA/"validation_y.npy")==m["validation"]["y_sha256"]
checks["test_labels_not_read"]=True
write_json(BASE/"instrumentos/contratos.json",dict(checks=checks,features=names,passed=all(checks.values())))
assert all(checks.values()),checks
print(checks,flush=True)

