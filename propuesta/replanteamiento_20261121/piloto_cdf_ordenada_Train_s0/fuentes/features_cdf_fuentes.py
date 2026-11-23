"""Boundary-conditioned historical CDFs and same-size geometry-derived controls."""
from base_replanteo import *
from recuperar_subtrayectos import metric_xy
GRID=np.arange(60,1800,15,dtype=np.float64)
GEOM_NAMES=['available','min_endpoint_error_km','weighted_error_km','error_std_km','ESS_fraction','max_weight',
 'full_mass','touch_mass','internal_mass','delta_ox','delta_oy','delta_dx','delta_dy','source_chord_ratio','source_chord_ratio_std']
def source_features(data,d,result):
 parent,first,last,scaled_time,logweight,error,full=result[:7]
 valid=parent>=0;safe=np.maximum(parent,0);offset=data['offsets'][safe]
 start=data['xy_m'][offset+first].astype(np.float64);end=data['xy_m'][offset+last].astype(np.float64)
 query=metric_xy(d);chord=np.sqrt(((end-start)**2).sum(-1));qlength=np.sqrt(((query[:,2:]-query[:,:2])**2).sum(-1))
 scale=np.clip(qlength[:,None]/np.maximum(chord,1),.5,2)
 reconstructed=15*(last.astype(int)-first.astype(int))*scale
 assert np.allclose(reconstructed[valid],scaled_time[valid],atol=.001,rtol=1e-6)
 lw=np.where(valid,logweight,-np.inf);maximum=lw.max(1);maximum[~np.isfinite(maximum)]=0
 weight=np.where(valid,np.exp(lw-maximum[:,None]),0);weight/=np.maximum(weight.sum(1,keepdims=True),1e-30)
 full=valid&full;touch=valid&~full&((first==0)|(last==data['parent_gps_n'][safe]-1));internal=valid&~full&~touch
 groups=[full,touch,internal];mass=np.column_stack([(weight*g).sum(1) for g in groups])
 err=np.where(valid,error,0).astype(float);mean=(weight*err).sum(1)
 std=np.sqrt(np.maximum(0,(weight*err**2).sum(1)-mean**2))
 delta=np.concatenate([start,end],-1)-query[:,None,:]
 ratio=chord/np.maximum(qlength[:,None],1);rmean=(weight*ratio).sum(1)
 rstd=np.sqrt(np.maximum(0,(weight*ratio**2).sum(1)-rmean**2))
 ess=np.where(valid.any(1),1/np.maximum((weight**2).sum(1),1e-30)/64,0)
 geom=np.column_stack([valid.any(1),np.minimum(np.where(valid,error,np.inf).min(1),1e6)/1000,mean/1000,std/1000,
 ess,weight.max(1),mass,(weight[...,None]*delta).sum(1)/1000,rmean,rstd]).astype(np.float32)
 assert geom.shape==(len(parent),15)
 # Geometrycontrol uses fixed25km/h, the inherited OD fallback speed, never observed source time.
 proxy=chord/(25/3.6)*scale
 cdfs=[]
 for time_values in [scaled_time,proxy]:
  comparison=time_values[...,None]<=GRID[None,None,:]
  cdfs.append(np.concatenate([(comparison*(weight*g)[...,None]).sum(1) for g in groups],1).astype(np.float32))
 assert cdfs[0].shape==(len(parent),348) and np.isfinite(geom).all() and all(np.isfinite(v).all() for v in cdfs)
 return geom,cdfs[0],cdfs[1]
