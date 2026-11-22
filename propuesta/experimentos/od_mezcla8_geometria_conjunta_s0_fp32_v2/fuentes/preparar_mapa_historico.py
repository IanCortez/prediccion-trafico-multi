from common import *
from features import ORIGIN
import requests,osmium,argparse
parser=argparse.ArgumentParser();parser.add_argument('--year',choices=['2014','2026'],default='2014');year=parser.parse_args().year
started=time.time();folder=BASE/'referencias/mapas';folder.mkdir(exist_ok=True);source=folder/f'portugal-{year[2:]}0101.osm.pbf';url=f'https://download.geofabrik.de/europe/portugal-{year[2:]}0101.osm.pbf'
if not source.exists():
    temporary=source.with_suffix('.pbf.partial')
    with requests.get(url,stream=True,timeout=(15,60)) as response:
        response.raise_for_status();expected=response.headers.get('Content-Length');received=0
        with temporary.open('wb') as f:
            for block in response.iter_content(1024*1024):f.write(block);received+=len(block)
        assert expected is None or received==int(expected)
    temporary.replace(source)
print(dict(stage='download',bytes=source.stat().st_size,seconds=time.time()-started),flush=True)
SPEED={'motorway':100,'motorway_link':60,'trunk':80,'trunk_link':50,'primary':60,'primary_link':40,'secondary':50,'secondary_link':35,'tertiary':40,'tertiary_link':30,'unclassified':30,'residential':30,'living_street':10,'service':20,'road':30}
class Roads(osmium.SimpleHandler):
    def __init__(self):super().__init__();self.coords={};self.edges={};self.ways={}
    def way(self,way):
        tag=dict(way.tags);highway=tag.get('highway')
        if highway not in SPEED or tag.get('access') in ('no','private') or tag.get('motor_vehicle')=='no':return
        points=[]
        for node in way.nodes:
            if not node.location.valid():points.append(None);continue
            lat=node.location.lat;lon=node.location.lon
            points.append((int(node.ref),lon,lat) if 41.09<=lat<=41.24 and -8.76<=lon<=-8.48 else None)
        if not any(p is not None for p in points):return
        try:speed=float(str(tag.get('maxspeed','')).split(';')[0].split()[0]);speed=max(5,min(speed,130))
        except (ValueError,IndexError):speed=SPEED[highway]
        one=tag.get('oneway','yes' if tag.get('junction')=='roundabout' or highway=='motorway' else 'no');used=0
        for a,b in zip(points[:-1],points[1:]):
            if a is None or b is None:continue
            u,lon1,lat1=a;v,lon2,lat2=b
            dx=(lon2-lon1)*111195*np.cos(np.deg2rad(ORIGIN[1]));dy=(lat2-lat1)*111195;length=float(np.hypot(dx,dy))
            if length<.01:continue
            self.coords[u]=(lon1,lat1);self.coords[v]=(lon2,lat2)
            directions=[(u,v)] if one in ('yes','1','true') else [(v,u)] if one=='-1' else [(u,v),(v,u)]
            for pair in directions:
                old=self.edges.get(pair,(float('inf'),float('inf')));self.edges[pair]=(min(old[0],length),min(old[1],length/(speed/3.6)))
            used+=1
        self.ways[highway]=self.ways.get(highway,0)+used
out=DATA/f'mapa_dirigido_{year}.npz';assert not out.exists(),out
handler=Roads();handler.apply_file(str(source),locations=True,idx='flex_mem')
ids=np.array(sorted(handler.coords),dtype=np.int64);lookup={int(node):i for i,node in enumerate(ids)};lonlat=np.array([handler.coords[int(node)] for node in ids]);xy=np.column_stack([(lonlat[:,0]-ORIGIN[0])*111195*np.cos(np.deg2rad(ORIGIN[1])),(lonlat[:,1]-ORIGIN[1])*111195]);pairs=list(handler.edges);u=np.array([lookup[p[0]] for p in pairs]);v=np.array([lookup[p[1]] for p in pairs]);length,seconds=np.array(list(handler.edges.values())).T
np.savez_compressed(out,xy=xy,lonlat=lonlat,u=u,v=v,length=length,seconds=seconds,node_ids=ids)
reader=osmium.io.Reader(str(source));header=reader.header();timestamp=header.get('osmosis_replication_timestamp');reader.close()
write_json(DATA/f'mapa_{year}_manifiesto.json',dict(source_url=url,source_sha256=sha(source),source_bytes=source.stat().st_size,archive_date=year+'-01-01',header_replication_timestamp=timestamp,nodes=len(ids),directed_arcs=len(u),way_segments=handler.ways,bbox=[41.09,41.24,-8.76,-8.48],graph_sha256=sha(out),license='OpenStreetMap contributors, ODbL; regional archive provided by Geofabrik',source_index='https://download.geofabrik.de/europe/portugal.html',test_labels_read=False,limitation='OSM state on archive date, not ground-truth road construction or traffic conditions. Added as a separate candidate; modern map files preserved.',seconds=time.time()-started))
print(dict(stage='graph',nodes=len(ids),arcs=len(u),timestamp=timestamp,seconds=time.time()-started),flush=True)

