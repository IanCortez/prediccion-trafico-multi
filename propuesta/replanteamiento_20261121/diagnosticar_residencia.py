"""Train-held-week diagnostic for factorization of movement residence and next direction."""
from base_replanteo import *
import math
start=time.time();access=deny_test();z=np.load(OUT/'datos_semi_markov/runs_train.npz');cells=z['cell'].astype(np.int32);dur=z['duration_n'].astype(np.int32);off=z['run_offsets'];trip=z['trip_train_index'];valid=np.unpackbits(z['outgoing_valid_bits'])[:len(cells)].astype(bool);tr=dict(np.load(DATA/'train_x.npz'));grid=read(OUT/'datos_semi_markov/protocolo.json');N=grid['N'];ny=grid['ny'];G=N*5*32
sx=111195*np.cos(np.deg2rad(41.16));dest=np.column_stack([(tr['d_lon'][trip]+8.62)*sx,(tr['d_lat'][trip]-41.16)*111195]);ts=tr['timestamp'][trip];held=((ts//(7*86400))%5)==0
def heading(dx,dy):return np.where(np.abs(dx)>=np.abs(dy),np.where(dx>=0,0,1),np.where(dy>=0,2,3)).astype(np.int32)
def chunk(first,last):
 ri=np.arange(first,last);ti=np.searchsorted(off,ri,side='right')-1;cur=cells[ri];nxt=cells[np.minimum(ri+1,len(cells)-1)];mask=valid[ri]&(ri<off[ti+1]-1)&(cur<N)&(nxt<N);ri=ri[mask];ti=ti[mask];cur=cur[mask];nxt=nxt[mask];prev=cells[np.maximum(ri-1,0)];ok=(ri>off[ti])&(prev<N)&valid[np.maximum(ri-1,0)];h=np.where(ok,heading(cur//ny-prev//ny,cur%ny-prev%ny),4);out=heading(nxt//ny-cur//ny,nxt%ny-cur%ny);xy=np.column_stack([grid['x0']+(cur//ny+.5)*200,grid['y0']+(cur%ny+.5)*200]);rel=dest[ti]-xy;angle=np.floor((np.arctan2(rel[:,1],rel[:,0])+math.pi)/(2*math.pi)*8).astype(np.int32).clip(0,7);distance=np.searchsorted([500,1500,4000],np.linalg.norm(rel,axis=1));co=angle*4+distance;g=(cur*5+h)*32+co;return g,h*32+co,out,dur[ri].clip(1,119)-1,held[ti],ts[ti]//86400
base=np.zeros((G,119),np.uint32);joint=np.zeros((G*4,119),np.uint32);global_c=np.zeros((5*32,119),np.uint32);fitn=0
for first in range(0,len(cells)-1,200000):
 g,co,a,y,hold,days=chunk(first,min(first+200000,len(cells)-1));fit=~hold;fitn+=int(fit.sum());np.add.at(base,(g[fit],y[fit]),1);np.add.at(joint,(g[fit]*4+a[fit],y[fit]),1);np.add.at(global_c,(co[fit],y[fit]),1)
nb=base.sum(1);nj=joint.sum(1);gp=(global_c.astype(float)+1)/(global_c.sum(1)[:,None]+119);n=0;lb=0.;lj=0.;day_sum={};day_n={}
for first in range(0,len(cells)-1,100000):
 g,co,a,y,hold,days=chunk(first,min(first+100000,len(cells)-1));g=g[hold];co=co[hold];a=a[hold];y=y[hold];days=days[hold];p=(base[g,y]+20*gp[co,y])/(nb[g]+20);pj=(joint[g*4+a,y]+20*p)/(nj[g*4+a]+20);eb=-np.log(p);ej=-np.log(pj);n+=len(g);lb+=eb.sum();lj+=ej.sum()
 for d in np.unique(days):
  m=days==d;day_sum[int(d)]=day_sum.get(int(d),0.)+float((ej[m]-eb[m]).sum());day_n[int(d)]=day_n.get(int(d),0)+int(m.sum())
days=sorted(day_n);s=np.array([day_sum[d] for d in days]);nn=np.array([day_n[d] for d in days]);ix=np.random.default_rng(1729).integers(len(days),size=(3000,len(days)));ci=np.quantile(s[ix].sum(1)/nn[ix].sum(1),[.025,.975]);report=dict(kind='Train-only held-week conditional residence likelihood; NOT duration MAE',fit_moves=fitn,held_moves=n,baseline_conditioning='Cell, incoming heading4/unknown, destination bearing8 and distance4',joint_conditioning='Same plus next outgoing heading4, as a candidate action to marginalize at query',baseline_NLL_nats=lb/n,joint_NLL_nats=lj/n,delta_nats=(lj-lb)/n,delta_CI95_by_day=ci,implication='Tests whether independence of next action and residence loses predictive information. Next actual action is NOT allowed as query input; only a jointly generated/marginalized action may condition waiting time.',Validation_or_Test_paths_decoded=False,Test_access=False,seconds=time.time()-start,code_sha256=sha(__file__))
write(OUT/'diagnostico_dependencia_residencia.json',report);print(json.dumps(report,default=lambda x:x.tolist()))

