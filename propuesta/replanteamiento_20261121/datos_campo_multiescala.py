"""Fixed primary cohort and Train-only constructor for fresh spatial-field models."""
from base_replanteo import *
import torch,rtdl_num_embeddings as rtdl
from inferencia import feature_input
def load_data(stats_from=None):
 rec=read(BEST/'resultado.json');tr,X,names=feature_input('train',rec)
 assert names==rec['features'] and names[:4]==['ox','oy','dx','dy'] and all(not n.startswith('call_') for n in names)
 eligible=np.load(OUT/'datos_semi_markov/runs_train.npz')['trip_train_index'];assert len(eligible)==1023485
 row=tr['row'][eligible];X=X[eligible].copy();y=np.load(DATA/'train_y.npy')[eligible].copy()
 va,V,vnames=feature_input('validation',rec);assert vnames==names
 v=np.load(DATA/'validation_y.npy');assert len(v)==127941
 for F in [X,V]:
  assert F[:,[0,2]].min()*1000>=-6500 and F[:,[0,2]].max()*1000<=6500
  assert F[:,[1,3]].min()*1000>=-3000 and F[:,[1,3]].max()*1000<=4000
 gt=dict(np.load(DATA/'geometria_train.npz'));pos=np.searchsorted(gt['row'],row);safe=np.minimum(pos,len(gt['row'])-1)
 mask=(pos<len(gt['row']))&(gt['row'][safe]==row)&(gt['X'][safe,-1]<=500)
 geometry=gt['X'][safe,1:15].copy();mask&=np.isfinite(geometry).all(1);geometry[~mask]=0
 if stats_from:
  ctor=torch.load(stats_from,map_location='cpu',weights_only=False);assert ctor['features']==names and ctor['Train_n']==len(X)
 else:
  keep=np.flatnonzero(np.ptp(X,axis=0)>1e-7);mean=X[:,keep].mean(0,dtype=np.float64).astype(np.float32);std=np.maximum(X[:,keep].std(0,dtype=np.float64),1e-5).astype(np.float32)
  z=(X[:,keep]-mean)/std;sample=np.random.default_rng(0).choice(len(X),min(200000,len(X)),replace=False)
  bins=rtdl.compute_bins(torch.from_numpy(np.vstack([z[sample],z.min(0),z.max(0)])),n_bins=48);del z
  anchor_indices=[names.index(n) for n in ['mm_length_gps_time_600s','mm_free_gps_time_600s','projected_free_freeflow_600s','od_km']]
  anchors=[]
  for col,scale in zip(anchor_indices,[600,600,600,1000]):
   logx=np.log(np.maximum(X[:,col].astype(float)*scale,.1));beta,intercept=np.polyfit(logx,np.log(y),1);intercept+=np.log(np.median(y/np.exp(intercept+beta*logx)));anchors.append([intercept,beta])
  short=[names.index('projected_length_shape_'+str(i)) for i in range(14)];fast=[names.index('projected_free_shape_'+str(i)) for i in range(14)]
  clocks=[names.index(n) for n in ['hour_sin','hour_cos','week_sin','week_cos','season_sin','season_cos','weekend','calendar_special']]
  ctor=dict(mean=mean,std=std,keep=keep,bins=bins,anchors=anchors,anchor_indices=anchor_indices,geometry_indices=np.array([short,fast,fast,[-1]*14]*2),clock_indices=clocks,features=names,Train_n=len(X),k=16,width=512)
 return rec,tr,eligible,row,X,y,geometry,mask,va,V,v,ctor
