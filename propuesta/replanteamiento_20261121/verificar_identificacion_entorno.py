"""Independent synthetic-generation check that the pilot can distinguish shared from independent effects."""
from base_replanteo import *
from verosimilitud_entorno import environment_log_likelihood as ll
import torch,math
torch.set_num_threads(4);torch.manual_seed(77);B,R=6000,8;nodes,weights=np.polynomial.hermite.hermgauss(4);nodes=torch.tensor(nodes,dtype=torch.float32);weights=torch.tensor(weights/math.sqrt(math.pi),dtype=torch.float32);logw=torch.zeros(B,R,1);base=torch.randn(B,R,1)*.2;valid=torch.ones(B,R,dtype=torch.bool);lags=torch.tensor([0]);true_sigma=.7;true_location=.1;results={}
for generating in ['shared','independent']:
 z=torch.multinomial(weights,B if generating=='shared' else B*R,replacement=True);eta=true_location+math.sqrt(2)*true_sigma*nodes[z];eta=eta[:,None] if generating=='shared' else eta.reshape(B,R);p=torch.sigmoid(base[:,:,0]-eta);u=torch.rand(B,R).clamp_min(1e-9);duration=(torch.log(u)/torch.log1p(-p)).floor().long()+1
 fit=torch.arange(4500);held=torch.arange(4500,B);scores={}
 for fitted in ['shared','independent']:
  bias=torch.nn.Parameter(torch.tensor(0.));raw=torch.nn.Parameter(torch.tensor(math.log(math.expm1(.15))));opt=torch.optim.Adam([bias,raw],lr=.04)
  for step in range(250):
   opt.zero_grad();location=2*bias.tanh();sigma=torch.nn.functional.softplus(raw).clamp_max(1.5);v=ll(logw[fit],base[fit],duration[fit],valid[fit],lags,location,sigma,nodes,weights,fitted);loss=-v.mean()/R;loss.backward();opt.step()
  with torch.no_grad():
   location=2*bias.tanh();sigma=torch.nn.functional.softplus(raw).clamp_max(1.5);v=ll(logw[held],base[held],duration[held],valid[held],lags,location,sigma,nodes,weights,fitted);scores[fitted]=dict(held_NLL_per_run=float(-v.mean()/R),location=float(location),sigma=float(sigma))
 assert abs(scores[generating]['sigma']-true_sigma)<.15 and abs(scores[generating]['location']-true_location)<.15
 other='independent' if generating=='shared' else 'shared'
 assert scores[generating]['held_NLL_per_run']<scores[other]['held_NLL_per_run']
 results[generating]=scores
assert not torch.cuda.is_initialized()
write(OUT/'verificacion_identificacion_entorno.json',dict(passed=True,synthetic_only=True,true_sigma=true_sigma,true_location=true_location,results=results,correct_dependence_selected_in_both_directions=True,correct_parameters_recovered=True,no_user_trip_data_used=True,no_GPU_used=True,not_a_travel_time_MAE_result=True,code_sha256=sha(__file__),likelihood_code_sha256=sha(OUT/'verosimilitud_entorno.py')))
print(json.dumps(results))

