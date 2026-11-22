from base_replanteo import *
from modelo_memoria_residual import ResidualMemory,stream_value
import torch
torch.set_num_threads(2);torch.manual_seed(3);m=ResidualMemory(7);m.eval();x=torch.randn(90,7);g=torch.randn(90,4);value=torch.randn(90);q,gate=m.encode(x[:5],g[:5]);k,_=m.encode(x,g);ref=m.scores(q,k).softmax(-1)@value
got=stream_value(q,k,value,chunk=23);assert torch.allclose(ref,got,atol=1e-6)
pred,w=m.correction(x[:1],g[:1],x,g,torch.tensor([0]),torch.arange(90),value);assert w[0,0]==0
changed=value.clone();changed[0]+=999;again,_=m.correction(x[:1],g[:1],x,g,torch.tensor([0]),torch.arange(90),changed);assert torch.equal(pred,again)
write(OUT/'verificacion_residual_cpu.json',dict(passed=True,streaming_max_error=float((ref-got).abs().max()),self_label_intervention_invariant=True,Test_access=False))

