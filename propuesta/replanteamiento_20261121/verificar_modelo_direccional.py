from base_replanteo import *
from modelo_direccional import DirectionalArrival
import torch
torch.set_num_threads(4);torch.manual_seed(0);ck=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False);grid=read(OUT/'datos_semi_markov/protocolo.json');graph=dict(np.load(OUT/'datos_semi_markov/grafo_grid.npz'));print(list(graph),flush=True);distance=np.load(OUT/'datos_semi_markov/distancias_grid.npy');model=DirectionalArrival(ck,grid,graph,distance,ck['anchors'][3]).eval();x=torch.zeros(2,216);x[:,ck['keep']]=torch.as_tensor(ck['mean']);x[0,:4]=torch.tensor([0.,0.,1.,1.]);x[0,6]=2**.5;x[1,:4]=torch.tensor([0.,0.,.01,.01]);x[1,6]=.02
with torch.no_grad():
 p,mass,p0,path=model.posterior(x);rev=model.posterior(x.flip(0))[0].flip(0)
assert torch.isfinite(p).all() and torch.min(p)>=0 and (p.sum(-1)-1).abs().max()<1e-5 and torch.all((mass>=0)&(mass<=1+1e-4));assert (p-rev).abs().max()<1e-6
write(OUT/'verificacion_modelo_direccional_cpu.json',dict(passed=True,probabilities_finite_and_normalized=True,finite_horizon_absorption_mass=mass.numpy(),batch_permutation_max_error=float((p-rev).abs().max()),same_cell_origin_destination_supported=True,query_inputs='Synthetic OD features only; unknown direction at origin, no target or route',trained=False,predictive_MAE_not_evaluated=True,Test_access=False,parameters=sum(p.numel() for p in model.parameters())))
print('Directional neural spatial-process probability and query invariance checks passed.',flush=True)

