from base_replanteo import *
from modelo_dir_fases import PhaseDirectionalArrival
import torch
torch.set_num_threads(4);torch.manual_seed(0);access=deny_test();ck=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False);grid=read(OUT/'datos_semi_markov/protocolo.json');graph=dict(np.load(OUT/'datos_semi_markov/grafo_grid.npz'));distance=np.load(OUT/'datos_semi_markov/distancias_grid.npy');counts=np.load(OUT/'prior_transiciones/conteos_fold.npz')['counts'];init=read(OUT/'inicializacion_fases.json');model=PhaseDirectionalArrival(ck,grid,graph,distance,ck['anchors'][3],counts,init).eval();x=torch.zeros(2,216);x[:,ck['keep']]=torch.as_tensor(ck['mean']);x[0,:4]=torch.tensor([0.,0.,1.,1.]);x[0,6]=2**.5;x[1,:4]=torch.tensor([0.,0.,.01,.01]);x[1,6]=.02
with torch.no_grad():
 p,mass,p0,path=model.posterior(x,excluded_fold=torch.tensor([0,1]));rev=model.posterior(x.flip(0),excluded_fold=torch.tensor([1,0]))[0].flip(0)
assert torch.isfinite(p).all() and (p.sum(-1)-1).abs().max()<1e-5 and torch.all((mass>=0)&(mass<=1.0001));assert (p-rev).abs().max()<1e-6
report=dict(passed=True,probability_normalization=True,query_permutation_max_error=float((p-rev).abs().max()),mass=mass.numpy(),same_cell_origin_destination=True,fold_exclusion_in_full_inference=True,query_inputs='Synthetic OD fields only, no target/point count/path',model_code_sha256=sha(OUT/'modelo_dir_fases.py'),kernel_code_sha256=sha(OUT/'nucleo_fases.py'),Test_access=False,trained=False)
write(OUT/'verificacion_modelo_fases_cpu.json',report);print(json.dumps(report,default=lambda z:z.tolist()))

