"""Linear OD-coordinate frame; supervised fitcoordinates only, no nearestpath projection."""
import numpy as np
QUERY={'o_lon','o_lat','d_lon','d_lat','timestamp'}
def metric_queries(queries):
 assert set(queries)==QUERY
 sx=111195*np.cos(np.deg2rad(41.16))
 return np.column_stack([(queries['o_lon']+8.62)*sx,(queries['o_lat']-41.16)*111195,
  (queries['d_lon']+8.62)*sx,(queries['d_lat']-41.16)*111195])
def fit_positions(points,query_xy):
 points=np.asarray(points,np.float64);xy=np.asarray(query_xy,np.float64)
 vector=xy[2:]-xy[:2];den=float(vector@vector);assert den>0
 relative=points-xy[:2]
 out=np.column_stack([(relative*vector).sum(1),vector[0]*relative[:,1]-vector[1]*relative[:,0]])/den
 endpoint_error_m=max(float(np.linalg.norm(points[0]-xy[:2])),float(np.linalg.norm(points[-1]-xy[2:])))
 out[0]=[0.,0.];out[-1]=[1.,0.]
 return out,endpoint_error_m
