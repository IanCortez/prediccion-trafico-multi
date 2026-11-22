from common import *
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from features import ORIGIN
import math
started=time.time()
OUT=DATA/"mapa_dirigido.npz"
SPEED={"motorway":100,"motorway_link":60,"trunk":80,"trunk_link":50,"primary":60,"primary_link":40,"secondary":50,"secondary_link":35,"tertiary":40,"tertiary_link":30,"unclassified":30,"residential":30,"living_street":10,"service":20,"road":30}
def metric(lon,lat):
    return np.column_stack(((np.asarray(lon)-ORIGIN[0])*111195*np.cos(np.deg2rad(ORIGIN[1])),(np.asarray(lat)-ORIGIN[1])*111195))
if not OUT.exists():
    coords={};ways={};sources=[]
    for path in sorted((ROOT/"tesis/osm/_overpass_cache").glob("*.json")):
        obj=json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(obj,dict):continue
        els=obj.get("elements",[])
        relevant=False
        for e in els:
            if e.get("type")=="node" and 41.09<=e.get("lat",0)<=41.24 and -8.76<=e.get("lon",0)<=-8.48:
                coords[e["id"]]=(e["lon"],e["lat"]);relevant=True
        if relevant:
            sources.append(dict(file=path.name,sha256=sha(path),osm_timestamp=obj.get("osm3s",{}).get("timestamp_osm_base")))
            for e in els:
                if e.get("type")=="way" and e.get("tags",{}).get("highway") in SPEED:ways[e["id"]]=e
    ids=np.array(sorted(coords));loc={j:i for i,j in enumerate(ids)};lonlat=np.array([coords[j] for j in ids]);xy=metric(lonlat[:,0],lonlat[:,1])
    edges={};way_counts={}
    for way in ways.values():
        tag=way["tags"];highway=tag["highway"]
        if tag.get("access") in ("no","private") or tag.get("motor_vehicle")=="no":continue
        try:speed=float(str(tag.get("maxspeed","")).split(";")[0].split()[0]);speed=max(5,min(speed,130))
        except (ValueError,IndexError):speed=SPEED[highway]
        one=tag.get("oneway","yes" if tag.get("junction")=="roundabout" or highway=="motorway" else "no")
        pairs=list(zip(way["nodes"][:-1],way["nodes"][1:]))
        used=0
        for u,v in pairs:
            if u not in loc or v not in loc:continue
            i,j=loc[u],loc[v];length=float(np.linalg.norm(xy[i]-xy[j]))
            if length<.01:continue
            directions=[(i,j)] if one in ("yes","1","true") else [(j,i)] if one=="-1" else [(i,j),(j,i)]
            for pair in directions:
                # Parallel ways retain minimum length and minimum free-flow time independently.
                old=edges.get(pair,(float("inf"),float("inf")))
                edges[pair]=(min(old[0],length),min(old[1],length/(speed/3.6)))
            used+=1
        way_counts[highway]=way_counts.get(highway,0)+used
    u,v=np.array(list(edges),dtype=np.int64).T;length,seconds=np.array(list(edges.values())).T
    # OSM segment geometry is retained as every consecutive original node.
    np.savez_compressed(OUT,xy=xy,lonlat=lonlat,u=u,v=v,length=length,seconds=seconds,node_ids=ids)
    write_json(DATA/"mapa_manifiesto.json",dict(sources=sources,nodes=len(ids),directed_edges=len(u),way_segments=way_counts,sha256=sha(OUT),method="Original Overpass consecutive-node geometry, direction and maxspeed; no label/trajectory fitting",limitation="Available OSM map is newer than 2013-2014 trips; no historical map claim",seconds=time.time()-started))
z=np.load(OUT);xy=z["xy"];n=len(xy)
active=np.unique(np.concatenate([z["u"],z["v"]]));tree=cKDTree(xy[active])
g=csr_matrix((z["length"],(z["u"],z["v"])),shape=(n,n))
gt=csr_matrix((z["seconds"],(z["u"],z["v"])),shape=(n,n))
# Batch all input-only queries to reuse each source search across partitions.
splits=("train","validation","test")
parts=[dict(np.load(DATA/(split+"_x.npz"))) for split in splits]
d={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
osnap,oi=tree.query(metric(d["o_lon"],d["o_lat"]));dsnap,di=tree.query(metric(d["d_lon"],d["d_lat"]))
oi=active[oi];di=active[di]
src,inv=np.unique(oi,return_inverse=True)
order=np.argsort(inv);bounds=np.searchsorted(inv[order],np.arange(len(src)+1))
partial=DATA/"rutas_parcial.npz"
if partial.exists():
    q=np.load(partial);assert np.array_equal(q["sources"],src)
    dist=q["dist"].copy();tt=q["tt"].copy();done=q["done"].copy()
else:
    dist=np.zeros(len(oi),dtype=np.float32);tt=dist.copy();done=np.zeros(len(src),dtype=bool)
for first in range(0,len(src),64):
    last=min(first+64,len(src))
    if done[first:last].all():continue
    ix=order[bounds[first]:bounds[last]]
    dl=dijkstra(g,directed=True,indices=src[first:last])
    dt=dijkstra(gt,directed=True,indices=src[first:last])
    dist[ix]=dl[inv[ix]-first,di[ix]]+osnap[ix]+dsnap[ix]
    tt[ix]=dt[inv[ix]-first,di[ix]]+(osnap[ix]+dsnap[ix])/(20/3.6)
    done[first:last]=True
    if first%512==0 or last==len(src):
        np.savez_compressed(partial,sources=src,done=done,dist=dist,tt=tt)
        print(dict(sources=f"{last}/{len(src)}",seconds=time.time()-started),flush=True)
valid=np.isfinite(dist)&np.isfinite(tt)&(osnap<500)&(dsnap<500)
dist=np.where(valid,dist,d["od_m"]).astype(np.float32)
tt=np.where(valid,tt,d["od_m"]/(25/3.6)).astype(np.float32)
X=np.column_stack([dist/1000,tt/600,np.log(np.maximum(dist/d["od_m"],.01)),np.log(np.maximum(tt*1000/np.maximum(dist,1),.01)),osnap/1000,dsnap/1000,valid.astype(float)]).astype(np.float32)
names=["network_shortest_km","network_fastest_time_600s","network_circuity_log","network_pace_log","o_snap_km","d_snap_km","route_available"]
start=0
for split,p in zip(splits,parts):
    end=start+len(p["row"]);dest=DATA/(split+"_rutas.npz")
    np.savez_compressed(dest,row=d["row"][start:end],X=X[start:end],names=np.array(names))
    print(dict(split=split,route_available=float(valid[start:end].mean()),shape=X[start:end].shape,seconds=time.time()-started),flush=True)
    start=end

