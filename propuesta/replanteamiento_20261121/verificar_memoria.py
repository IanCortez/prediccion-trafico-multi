"""CPU checks of exact kernel aggregation, query permutation and self exclusion."""
from base_replanteo import *
from modelo_memoria import LearnedMemory,stream_distribution
import torch
from torch import nn
torch.set_num_threads(4);torch.manual_seed(1729);model=LearnedMemory(206).eval();q=torch.randn(11,36);k=torch.randn(153,36);labels=torch.arange(153)%116;hot=nn.functional.one_hot(labels,116).float();prior=hot.mean(0);ref=.99*(model.scores(q,k).softmax(-1)@hot)+.01*prior;p=stream_distribution(q,k,hot,prior,chunk=37);reverse=stream_distribution(q.flip(0),k,hot,prior,chunk=37).flip(0);error=float((p-ref).abs().max());assert error<2e-6 and (p.sum(-1)-1).abs().max()<1e-6 and torch.allclose(p,reverse,rtol=0,atol=2e-7)
x=torch.randn(20,206);geo=torch.randn(20,4);w=model.neighborhood(x[:1],geo[:1],x,geo,torch.tensor([0]),torch.arange(20));y=torch.arange(20,dtype=torch.float32);a=w@y;y[0]=10000;b=w@y;assert torch.equal(a,b) and w[0,0]==0
write(OUT/'verificacion_memoria_cpu.json',dict(passed=True,streaming_full_reference_max_error=error,query_permutation_invariant=True,self_label_intervention_invariant=True,probabilities_normalized=True,Test_access=False))
print('Memory kernel checks passed; full aggregation equals streamed aggregation.')
