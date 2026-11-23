"""Explicit historical-estimator view; raw departure timestamp is never altered."""
import numpy as np,pandas as pd
from consulta_od import od_records
from consulta_mapa2014 import resources,ORIGIN,project_candidates,projected_route_features
from numba import set_num_threads
MOVEMENT_NAMES=['mm_'+kind+'_'+v for kind in ['length','free'] for v in ['gps_time_600s','state_support','link_support','congestion_log','pace_log']]
STATIC_NAMES=[name for kind in ['length','free'] for name in (['projected_'+kind+'_shape_'+str(i) for i in range(14)]+['projected_'+kind+'_'+v for v in ['polyline_km','circuity_log','turn_abs_pi','turn_signed_pi','sharp_turn_count','max_turn_pi','available','freeflow_600s','o_projection_km','d_projection_km','attachment_gap','reachable_fraction']])]
def prepare(items):
 d=od_records(items);set_num_threads(2)
 raw,comp,sample_edge,tree,routes,profiles=resources()
 def metric(tag):return np.column_stack([(d[tag+'_lon']-ORIGIN[0])*111195*np.cos(np.deg2rad(ORIGIN[1])),(d[tag+'_lat']-ORIGIN[1])*111195])
 origin=metric('o');destination=metric('d');projected=[]
 for xy in [origin,destination]:
  _,near=tree.query(xy,k=32,workers=2)
  projected.extend(project_candidates(xy,near,sample_edge,raw['xy'],raw['u'],raw['v']))
 local=pd.to_datetime(d['timestamp'],unit='s',utc=True).tz_convert('Europe/Lisbon')
 clock=(np.asarray(local.dayofweek)*24+np.asarray(local.hour)).astype(np.int32)
 return dict(d=d,raw=raw,routes=routes,profiles=profiles,origin=origin,destination=destination,projected=projected,clock=clock)
def history_features(prepared,held_fold):
 assert held_fold in range(6)
 d=prepared['d'];raw=prepared['raw'];oe,of,od,de,df,dd=prepared['projected']
 profile=(held_fold*168+prepared['clock']).astype(np.int32)
 ss=[];mm=[]
 for kind in ['length','free']:
  D,conn,routing=prepared['routes'][kind]
  S,M,error=projected_route_features(prepared['origin'],prepared['destination'],d['od_m'],oe,of,od,de,df,dd,profile,raw['u'],raw['v'],raw['length'],raw['seconds'],raw['xy'],raw['seconds'] if kind=='free' else raw['length'],D,conn,routing,*prepared['profiles'],kind=='free')
  assert float(error.max())<.1 and np.isfinite(S).all() and np.isfinite(M).all()
  ss.append(S);mm.append(M)
 return np.column_stack(mm),np.column_stack(ss)
