"""Rawfivefield publicpath and fit-only targetprojection. No queryN/T/path argument."""
import numpy as np
QUERY={'o_lon','o_lat','d_lon','d_lat','timestamp'}
def public_polyline(queries,features,names):
 assert set(queries)==QUERY
 sx=111195*np.cos(np.deg2rad(41.16))
 origin=np.column_stack([(queries['o_lon']+8.62)*sx,(queries['o_lat']-41.16)*111195])
 dest=np.column_stack([(queries['d_lon']+8.62)*sx,(queries['d_lat']-41.16)*111195])
 fraction=np.arange(1,8)/8.
 interior=origin[:,None]+fraction[None,:,None]*(dest-origin)[:,None]
 slots=[names.index('projected_free_shape_'+str(i)) for i in range(14)]
 valid=features[:,names.index('projected_free_available')]>.5
 residual=features[:,slots].reshape(-1,7,2).astype(np.float64)*1000
 interior+=np.where(valid[:,None,None],residual,0.)
 path=np.concatenate([origin[:,None],interior,dest[:,None]],1)
 assert np.isfinite(path).all()
 return path,valid
def project_fit_positions(points,public_path):
 points=np.asarray(points,np.float64);path=np.asarray(public_path,np.float64)
 segment=np.diff(path,axis=0);length=np.sqrt((segment**2).sum(1))
 prefix=np.r_[0,np.cumsum(length)];assert prefix[-1]>0
 fraction=np.clip(((points[:,None]-path[None,:-1])*segment[None]).sum(-1)/np.maximum(length**2,1e-18),0,1)
 nearest=path[None,:-1]+fraction[...,None]*segment[None]
 error=((points[:,None]-nearest)**2).sum(-1)
 chosen=error.argmin(1);i=np.arange(len(points))
 progress=(prefix[chosen]+fraction[i,chosen]*length[chosen])/prefix[-1]
 # These are private fit-target endpointsemantics, never predictioninput.
 progress[0]=0.;progress[-1]=1.
 quantized=np.rint(progress*65535).astype(np.uint16)
 return quantized,np.sqrt(error[i,chosen])
