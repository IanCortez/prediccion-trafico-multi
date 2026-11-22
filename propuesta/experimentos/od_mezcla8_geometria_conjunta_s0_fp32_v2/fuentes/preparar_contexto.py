from common import *
from features import ORIGIN
from scipy.spatial import cKDTree
sys.path.insert(0,str(ROOT/"codigo-ian/src"))
import urban_context as uc
started=time.time()
gridstep=25;xs=np.arange(-7000,7000,gridstep)+gridstep/2;ys=np.arange(-3000,4000,gridstep)+gridstep/2
gx,gy=np.meshgrid(xs,ys,indexing="ij");grid=np.column_stack([gx.ravel(),gy.ravel()])
def xy(rows):
    return np.array([[(r["lon"]-ORIGIN[0])*111195*np.cos(np.deg2rad(ORIGIN[1])),(r["lat"]-ORIGIN[1])*111195] for r in rows])
poi=uc._read_points(ROOT/"tesis/osm/porto_pois.geojson");roads=uc._read_points(ROOT/"tesis/osm/porto_roads.geojson")
classes=np.array([uc._classify_poi(p) for p in poi])
pcoords=xy(poi);rcoords=xy(roads)
columns=[];names=[]
def counts(coords,tag,radius):
    if len(coords):v=cKDTree(coords).query_ball_point(grid,radius,return_length=True,workers=4)
    else:v=np.zeros(len(grid))
    columns.append(np.log1p(v).astype(np.float32));names.append(f"{tag}_r{radius}_logcount")
for radius in (100,300,800):
    for cls in uc.POI_CLASSES:counts(pcoords[classes==cls],"poi_"+cls,radius)
    counts(rcoords[np.array([int(p.get("street_count",0))>=3 for p in roads])],"intersections",radius)
    counts(rcoords[np.array([bool(p.get("signal",False)) for p in roads])],"signals",radius)
    for cls in uc.ROAD_CLASSES:
        mask=np.array([uc.OSM_HIGHWAY_TO_CLASS.get(p.get("highway",""))==cls for p in roads])
        counts(rcoords[mask],"road_"+cls,radius)
values=np.column_stack(columns)
np.savez_compressed(DATA/"contexto_rejilla25m.npz",features=values,names=np.array(names),gridstep=gridstep,x0=-7000,y0=-3000,nx=len(xs),ny=len(ys))
for split in ("train","validation","test"):
    d=np.load(DATA/(split+"_x.npz"));parts=[]
    for tag in ("o","d"):
        x=(d[tag+"_lon"]-ORIGIN[0])*111195*np.cos(np.deg2rad(ORIGIN[1]));y=(d[tag+"_lat"]-ORIGIN[1])*111195
        i=np.clip(np.floor((x+7000)/gridstep).astype(int),0,len(xs)-1)
        j=np.clip(np.floor((y+3000)/gridstep).astype(int),0,len(ys)-1)
        parts.append(values[i*len(ys)+j])
    np.savez_compressed(DATA/(split+"_contexto.npz"),row=d["row"],X=np.column_stack(parts),names=np.array(["o_"+n for n in names]+["d_"+n for n in names]))
write_json(DATA/"contexto_manifiesto.json",dict(poi_source_sha256=sha(ROOT/"tesis/osm/porto_pois.geojson"),road_source_sha256=sha(ROOT/"tesis/osm/porto_roads.geojson"),grid_sha256=sha(DATA/"contexto_rejilla25m.npz"),grid_m=25,radii_m=[100,300,800],n_features=2*len(names),poi_n=len(poi),road_n=len(roads),method="Fixed geographic grid; log1p count per radius, physical scale retained, no ranks or trip-dependent normalization",seconds=time.time()-started))
print(dict(shape=values.shape,seconds=time.time()-started),flush=True)

