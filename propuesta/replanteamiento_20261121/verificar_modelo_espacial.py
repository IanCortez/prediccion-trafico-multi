"""CPU smoke check for the neural spatial-process prototype; no predictive claims."""
from base_replanteo import *
from modelo_espacial import SpatialArrival
import torch

torch.set_num_threads(4);torch.manual_seed(0);ck=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False);grid=read(OUT/'datos_semi_markov/protocolo.json');graph=dict(np.load(OUT/'datos_semi_markov/grafo_grid.npz'));distance=np.load(OUT/'datos_semi_markov/distancias_grid.npy');model=SpatialArrival(ck,grid,graph,distance,ck['anchors'][3]).eval();x=torch.zeros(1,216);x[:,ck['keep']]=torch.as_tensor(ck['mean']);x[0,:4]=torch.tensor([0.,0.,1.,1.]);x[0,6]=2**.5
with torch.no_grad():
 p,mass,p0,path=model.posterior(x)
assert torch.isfinite(p).all() and torch.min(p)>=0 and abs(float(p.sum())-1)<1e-5 and 0<=float(mass)<=1+1e-4
write(OUT/'verificacion_modelo_espacial_cpu.json',dict(passed=True,probabilities_finite_and_normalized=True,finite_horizon_absorption_mass=float(mass),query_inputs='Synthetic OD features only, no target or route',trained=False,predictive_MAE_not_evaluated=True,Test_access=False,parameters=sum(p.numel() for p in model.parameters())))
print('Neural spatial-process probability check passed.',flush=True)
