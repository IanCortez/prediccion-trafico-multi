from common import *
from scipy.sparse import csr_matrix
import osmium
started=time.time();source=BASE/'referencias/mapas/portugal-140101.osm.pbf';raw=dict(np.load(DATA/'mapa_dirigido_2014.npz'));lookup={int(v):i for i,v in enumerate(raw['node_ids'])};arc_lookup={(int(a),int(b)):i for i,(a,b) in enumerate(zip(raw['u'],raw['v']))}
SPEED={'motorway':100,'motorway_link':60,'trunk':80,'trunk_link':50,'primary':60,'primary_link':40,'secondary':50,'secondary_link':35,'tertiary':40,'tertiary_link':30,'unclassified':30,'residential':30,'living_street':10,'service':20,'road':30};classes=list(SPEED);road_class=np.full(len(raw['u']),255,np.uint8);lane=np.zeros(len(raw['u']),np.float32);flags=np.zeros((len(raw['u']),3),np.uint8);node_control=np.zeros((len(raw['xy']),4),np.uint8);best=np.full(len(raw['u']),np.inf)
class Attributes(osmium.SimpleHandler):
    def node(self,node):
        i=lookup.get(int(node.id),-1)
        if i<0:return
        t=dict(node.tags);node_control[i]=[t.get('highway')=='traffic_signals' or t.get('crossing')=='traffic_signals',t.get('highway')=='stop',t.get('highway')=='give_way',t.get('highway')=='crossing' or 'crossing' in t]
    def way(self,way):
        t=dict(way.tags);h=t.get('highway')
        if h not in SPEED or t.get('access') in ('no','private') or t.get('motor_vehicle')=='no':return
        ns=[lookup.get(int(n.ref),-1) for n in way.nodes]
        if not any(n>=0 for n in ns):return
        try:speed=float(str(t.get('maxspeed','')).split(';')[0].split()[0]);speed=max(5,min(speed,130))
        except (ValueError,IndexError):speed=SPEED[h]
        try:lanes=max(0,min(float(str(t.get('lanes','')).split(';')[0]),10))
        except ValueError:lanes=0.
        one=t.get('oneway','yes' if t.get('junction')=='roundabout' or h=='motorway' else 'no');fc=[t.get('bridge','no') not in ('no','0','false'),t.get('tunnel','no') not in ('no','0','false'),t.get('junction')=='roundabout']
        for a,b in zip(ns[:-1],ns[1:]):
            if a<0 or b<0:continue
            directions=[(a,b)] if one in ('yes','1','true') else [(b,a)] if one=='-1' else [(a,b),(b,a)]
            for pair in directions:
                e=arc_lookup.get(pair,-1)
                if e<0:continue
                cost=raw['length'][e]/(speed/3.6)
                if cost<best[e]:best[e]=cost;road_class[e]=classes.index(h);lane[e]=lanes;flags[e]=fc
Attributes().apply_file(str(source));assert np.all(road_class<255);difference=float(np.max(np.abs(best-raw['seconds'])));assert difference<1e-6,difference
g=csr_matrix((np.ones(len(raw['u'])),(raw['u'],raw['v'])),shape=(len(raw['xy']),len(raw['xy'])));degree=np.diff((g+g.T).indptr);path=DATA/'atributos_mapa2014.npz';assert not path.exists();np.savez_compressed(path,road_class=road_class,classes=np.array(classes),lanes=lane,flags=flags,node_control=node_control,degree=degree)
report=dict(source_sha256=sha(source),graph_sha256=sha(DATA/'mapa_dirigido_2014.npz'),output_sha256=sha(path),arcs=len(road_class),node_control_counts=node_control.sum(0),node_controls=['signals','stop','give_way','crossing'],edge_flags=['bridge','tunnel','roundabout'],lanes_known_fraction=float(np.mean(lane>0)),max_freeflow_reconstruction_error_s=difference,selection='Attributes of the admissible way with minimum freeflow time for each directed raw node pair, matching the frozen graph construction.',limitation='Missing OSM traffic-control or lane tags do not prove absence. Static archive metadata, not observed traffic.',test_labels_read=False,seconds=time.time()-started);write_json(DATA/'atributos_mapa2014_manifiesto.json',report);print(json.dumps(report,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)),flush=True)
