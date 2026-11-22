from common import *
from features import ORIGIN
started=time.time();coords={};nodes={};ways={}
for path in sorted((ROOT/"tesis/osm/_overpass_cache").glob("*.json")):
    obj=json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(obj,dict):continue
    relevant=False
    for e in obj.get("elements",[]):
        if e.get("type")=="node" and 41.09<=e.get("lat",0)<=41.24 and -8.76<=e.get("lon",0)<=-8.48:
            coords[e["id"]]=(e["lon"],e["lat"]);nodes[e["id"]]=e;relevant=True
    if relevant:
        for e in obj.get("elements",[]):
            if e.get("type")=="way":ways[e["id"]]=e
keys=("amenity","shop","tourism","leisure","office","railway","public_transport","landuse")
texts=[];text_ids={};locations=[];indices=[];kinds=[];weights=[]
def text_id(s):
    if s not in text_ids:text_ids[s]=len(texts);texts.append(s)
    return text_ids[s]
def metric(p):
    return ((p[0]-ORIGIN[0])*111195*np.cos(np.deg2rad(ORIGIN[1])),(p[1]-ORIGIN[1])*111195)
def add(p,text,kind,weight=1):
    locations.append(metric(p));indices.append(text_id(text));kinds.append(kind);weights.append(weight)
def poi(p,tag):
    selected=[k+" = "+str(tag[k]) for k in keys if k in tag]
    if selected:
        if tag.get("name"):selected.append("name = "+tag["name"])
        add(p,"; ".join(selected),0)
for e in nodes.values():poi(coords[e["id"]],e.get("tags",{}))
for e in ways.values():
    tag=e.get("tags",{});points=[coords[i] for i in e.get("nodes",[]) if i in coords]
    if not points:continue
    poi(np.mean(points,axis=0),tag)
    if "highway" not in tag:continue
    label="road type = "+str(tag["highway"])
    if tag.get("name"):label+="; name = "+tag["name"]
    if tag.get("maxspeed"):label+="; speed limit = "+str(tag["maxspeed"])
    for u,v in zip(e["nodes"][:-1],e["nodes"][1:]):
        if u not in coords or v not in coords:continue
        a=np.asarray(coords[u]);b=np.asarray(coords[v]);distance=float(np.linalg.norm(np.asarray(metric(a))-metric(b)))
        # Sample long original segments at <=50m spacing, with physical length weights.
        steps=max(1,int(np.ceil(distance/50)))
        for j in range(steps):
            f=(j+.5)/steps;add(a*(1-f)+b*f,label,1,max(distance/steps,1))
write_json(DATA/"textos_urbanos.json",texts)
np.savez_compressed(DATA/"entidades_textuales.npz",xy=np.array(locations,dtype=np.float32),text_idx=np.array(indices,dtype=np.int32),kind=np.array(kinds,dtype=np.int8),weight=np.array(weights,dtype=np.float32))
write_json(DATA/"entidades_textuales_manifiesto.json",dict(unique_texts=len(texts),point_entities=len(locations),poi_points=kinds.count(0),road_points=kinds.count(1),fields="OSM type, name, highway/maxspeed; no personal contacts or addresses",source="Existing local Overpass cache; no added trajectory datasets",sha256=sha(DATA/"entidades_textuales.npz"),seconds=time.time()-started))
print(dict(texts=len(texts),points=len(locations),seconds=time.time()-started),flush=True)

