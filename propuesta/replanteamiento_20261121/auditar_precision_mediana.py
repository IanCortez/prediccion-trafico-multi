"""Outcome-free mixture-root stress test; no training module is changed."""
from base_replanteo import *
import torch,math
from cuantiles_mezcla_implicita import mixture_quantile
torch.set_num_threads(2);rows=[]
for scale in [.06,.1,.25,.5]:
 loc=torch.tensor([[math.log(100),math.log(900)]],dtype=torch.float32)
 sigma=torch.full_like(loc,scale);w=torch.full_like(loc,-math.log(2))
 f32=float(mixture_quantile(loc,sigma,w))
 # Independent symmetry: equallyweighted/equal-logscale modes have geometricmean median.
 f64=float(mixture_quantile(loc.double(),sigma.double(),w.double().log_softmax(-1)))
 rows.append(dict(sigma=scale,float32_s=f32,float64_s=f64,reference_geometric_mean_s=300.,absolute_difference_s=abs(f32-f64)))
result=dict(complete=True,scope='Unsupervised adversarial root conditioning; not a predictive experiment.',rows=rows,
 worst_difference_s=max(r['absolute_difference_s'] for r in rows),
 action='Inspect numerical conditioning of actual frozen posteriors before interpreting their MAE; do not edit active training modules.',Test=False)
write(OUT/'auditoria_precision_mediana.json',result);print(json.dumps(result))
