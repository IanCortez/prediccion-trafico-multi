"""Fit compact residence-family initialization using Train histogram only."""
from base_replanteo import *
from nucleo_fases import geometric_wait_pmf
import torch
torch.set_num_threads(4);access=deny_test();z=np.load(OUT/'datos_semi_markov/runs_train.npz');valid=np.unpackbits(z['outgoing_valid_bits'])[:len(z['cell'])].astype(bool);dur=z['duration_n'][valid].astype(np.int64);assert dur.min()>=1 and dur.max()<=119;hist=np.bincount(dur-1,minlength=119);freq=torch.tensor(hist/hist.sum(),dtype=torch.float64);lags=torch.tensor([0,0,1,1,2,2,3,3]);lw=torch.nn.Parameter(torch.zeros(8,dtype=torch.float64));lh=torch.nn.Parameter(torch.logit(torch.tensor([.65,.1]*4,dtype=torch.float64)));opt=torch.optim.Adam([lw,lh],lr=.03);start=time.time()
for step in range(2000):
 opt.zero_grad();w=lw.softmax(-1);h=lh.sigmoid().clamp(.001,.999);p=geometric_wait_pmf(w,h,lags,120);loss=-(freq*p.clamp_min(1e-30).log()).sum();loss.backward();opt.step()
with torch.no_grad():
 w=lw.softmax(-1);h=lh.sigmoid().clamp(.001,.999);p=geometric_wait_pmf(w,h,lags,120);entropy=float(-(freq[freq>0]*freq[freq>0].log()).sum());nll=float(-(freq*p.clamp_min(1e-30).log()).sum());mass=float(p.sum());mean=float((w*(lags+1/h)).sum())
report=dict(kind='Train-only global residence-family initialization; not a predictor MAE',n=len(dur),lags=lags.numpy(),weights=w.numpy(),hazards=h.numpy(),empirical_entropy_nats=entropy,fitted_NLL_nats=nll,excess_NLL_nats=nll-entropy,fitted_mass_within119=mass,empirical_mean_n=float((hist*np.arange(1,120)).sum()/hist.sum()),unbounded_model_mean_n=mean,Train_histogram=hist,iterations=2000,seconds=time.time()-start,Test_access=False,code_sha256=sha(__file__),kernel_code_sha256=sha(OUT/'nucleo_fases.py'))
write(OUT/'inicializacion_fases.json',report);print(json.dumps({k:report[k] for k in ['n','empirical_entropy_nats','fitted_NLL_nats','excess_NLL_nats','fitted_mass_within119','empirical_mean_n','unbounded_model_mean_n','seconds']}))

