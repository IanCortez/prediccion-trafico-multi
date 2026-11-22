"""Compact TRAIN spatial run targets for a semi-Markov OD time model."""
from base_replanteo import *
import csv,math
from array import array
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def main():
 start=time.time();run=OUT/'datos_semi_markov';run.mkdir(exist_ok=False);access=deny_test();cell=200.;margin=1000.;manifest=read(DATA/'manifiesto.json');lat0,lat1,lon0,lon1=manifest['bbox'];sx=111195*np.cos(np.deg2rad(41.16));x0=math.floor((lon0+8.62)*sx/cell)*cell-margin;y0=math.floor((lat0-41.16)*111195/cell)*cell-margin;x1=math.ceil((lon1+8.62)*sx/cell)*cell+margin;y1=math.ceil((lat1-41.16)*111195/cell)*cell+margin;nx=int(round((x1-x0)/cell));ny=int(round((y1-y0)/cell));N=nx*ny;assert N<65535;unknown=65535;offsets48=np.array([(a,b) for a in range(-3,4) for b in range(-3,4) if a or b],np.int32);xx=np.repeat(np.arange(nx),ny);yy=np.tile(np.arange(ny),nx);centers=np.column_stack([x0+(xx+.5)*cell,y0+(yy+.5)*cell]);neighbors=np.full((N,48),-1,np.int32)
 for k,(dx,dy) in enumerate(offsets48):
  a=xx+dx;b=yy+dy;ok=(a>=0)&(a<nx)&(b>=0)&(b<ny);neighbors[ok,k]=a[ok]*ny+b[ok]
 def encode(xy):
  a=np.floor((xy[:,0]-x0)/cell).astype(np.int64);b=np.floor((xy[:,1]-y0)/cell).astype(np.int64);ok=(a>=0)&(a<nx)&(b>=0)&(b<ny);return np.where(ok,a*ny+b,unknown).astype(np.uint16)
 tr=dict(np.load(DATA/'train_x.npz'));y=np.load(DATA/'train_y.npy');eligible=(tr['timestamp']+y)<manifest['validation']['start_timestamp'];lookup=np.full(1710670,-1,np.int32);lookup[tr['row'][eligible]]=np.flatnonzero(eligible);packed_cell=array('H');packed_duration=array('B');packed_enter=array('B');packed_valid=array('B');trip_indices=[];run_offsets=[0];observed=np.zeros((N,48),np.uint32);outside=0;point_n=0;seen=0;csv.field_size_limit(2**31-1)
 write(run/'protocolo.json',dict(cell_m=cell,margin_m=margin,nx=nx,ny=ny,N=N,x0=x0,y0=y0,source='Only Train GPS; arrival before first Validation departure',neighbor_radius=3,max_step_m=500,minimum_observed_support=5,public_reach_distance_m=800,query_route_is_input=False,Test_access=False))
 with (ROOT/'tesis/porto/porto/train.csv').open(encoding='utf-8-sig',newline='') as f:
  for row,r in enumerate(csv.DictReader(f)):
   j=lookup[row]
   if j<0:continue
   p=np.asarray(json.loads(r['POLYLINE']),np.float64);assert (len(p)-1)*15==y[j];xy=np.column_stack([(p[:,0]+8.62)*sx,(p[:,1]-41.16)*111195]);ids=encode(xy);starts=np.r_[0,np.flatnonzero(ids[1:]!=ids[:-1])+1];states=ids[starts];duration=np.diff(np.r_[starts,len(p)-1]);assert duration.sum()*15==y[j] and np.all(duration[:-1]>=1) and np.all(duration<=119);valid=np.zeros(len(states),bool)
   if len(states)>1:
    s=states[:-1].astype(np.int64);d=states[1:].astype(np.int64);dx=d//ny-s//ny;dy=d%ny-s%ny;steps=np.sqrt((np.diff(xy,axis=0)**2).sum(1))[starts[1:]-1];ok=(s!=unknown)&(d!=unknown)&(np.abs(dx)<=3)&(np.abs(dy)<=3)&(steps<=500);code=(dx+3)*7+(dy+3);action=code-(code>24);ok&=code!=24;valid[:-1]=ok;np.add.at(observed,(s[ok],action[ok]),1)
   packed_cell.frombytes(states.astype(np.uint16).tobytes());packed_duration.frombytes(duration.astype(np.uint8).tobytes());packed_enter.frombytes(starts.astype(np.uint8).tobytes());packed_valid.frombytes(valid.astype(np.uint8).tobytes());run_offsets.append(run_offsets[-1]+len(states));trip_indices.append(j);outside+=int((ids==unknown).sum());point_n+=len(ids);seen+=1
   if seen%200000==0:print(json.dumps(dict(trips=seen,runs=run_offsets[-1],seconds=time.time()-start)),flush=True)
 assert seen==eligible.sum();cells=np.frombuffer(packed_cell,dtype=np.uint16);dur=np.frombuffer(packed_duration,dtype=np.uint8);enter=np.frombuffer(packed_enter,dtype=np.uint8);valid=np.frombuffer(packed_valid,dtype=np.uint8).astype(bool);np.savez_compressed(run/'runs_train.npz',cell=cells,duration_n=dur,enter_n=enter,outgoing_valid_bits=np.packbits(valid),run_offsets=np.array(run_offsets,np.int64),trip_train_index=np.array(trip_indices,np.int32))
 raw=dict(np.load(DATA/'mapa_dirigido_2014.npz'));steps=np.maximum(1,np.ceil(raw['length']/50).astype(np.int32));counts=steps+1;edge=np.repeat(np.arange(len(steps)),counts);pos=np.arange(len(edge))-np.repeat(np.cumsum(counts)-counts,counts);a=raw['xy'][raw['u'][edge]];b=raw['xy'][raw['v'][edge]];f=pos/steps[edge];ids=encode(a*(1-f[:,None])+b*f[:,None]);ok=(edge[1:]==edge[:-1])&(ids[1:]!=ids[:-1])&(ids[1:]!=unknown)&(ids[:-1]!=unknown);keys=np.unique(ids[:-1][ok].astype(np.int64)*N+ids[1:][ok]);u=keys//N;v=keys%N;cost=np.sqrt(((centers[u]-centers[v])**2).sum(1));public=csr_matrix((cost,(u,v)),shape=(N,N));D=dijkstra(public,directed=True).astype(np.float32);public_mask=np.zeros((N,48),bool)
 for k in range(48):
  ok=neighbors[:,k]>=0;public_mask[ok,k]=D[np.flatnonzero(ok),neighbors[ok,k]]<=800
 allowed=(public_mask|(observed>=5))&(neighbors>=0);sources,actions=np.nonzero(allowed);targets=neighbors[sources,actions];cost=np.sqrt(((centers[sources]-centers[targets])**2).sum(1));graph=csr_matrix((cost,(sources,targets)),shape=(N,N));dist=dijkstra(graph,directed=True).astype(np.float32);np.save(run/'distancias_grid.npy',dist);np.savez_compressed(run/'grafo_grid.npz',neighbors=neighbors,allowed=allowed,observed=observed,centers=centers,offsets=offsets48)
 result=dict(trips=seen,runs=len(cells),GPS_points=point_n,outside_points=outside,grid_nodes=N,allowed_arcs=int(allowed.sum()),observed_step_coverage=float(observed[allowed].sum()/max(observed.sum(),1)),duration_partition_exact=True,first_run_at_elapsed0=True,terminal_run_duration_can_be0=True,all_training_trips_with_arrival_before_validation_retained=True,validation_and_test_paths_decoded=False,Test_access=False,seconds=time.time()-start,code_sha256=sha(__file__),outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [run/'runs_train.npz',run/'grafo_grid.npz',run/'distancias_grid.npy']},actual_data_files_accessed=sorted(access))
 write(run/'resultado.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
