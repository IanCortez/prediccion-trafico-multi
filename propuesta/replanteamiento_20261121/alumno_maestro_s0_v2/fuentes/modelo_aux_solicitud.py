"""Auxiliary request-type likelihood for route-duration components; OD-only forward."""
import torch
from torch import nn
class RequestAuxiliary(nn.Module):
    def __init__(self,duration):
        super().__init__();self.duration=duration
        self.request_head=nn.Linear(512,8*3)
        nn.init.zeros_(self.request_head.weight);nn.init.zeros_(self.request_head.bias)
    def forward(self,x):
        hidden=[]
        hook=self.duration.backbone.output.register_forward_pre_hook(lambda module,args:hidden.append(args[0]))
        try:out=self.duration(x)
        finally:hook.remove()
        assert len(hidden)==1
        log_type=self.request_head(hidden[0]).reshape(len(x),-1,8,3).log_softmax(-1)
        return out,log_type
    @staticmethod
    def request_score(output,log_type,target_seconds,request_label):
        # Actual type and time enter only the training objective, never forward.
        logp,weights,_,_,components=output[:5]
        label=((target_seconds-60)/15).long()
        B,K,J=weights.shape
        time_lp=components.gather(-1,label[:,None,None,None].expand(B,K,J,1))[...,0]
        marginal=logp.gather(-1,label[:,None,None].expand(B,K,1))[...,0]
        posterior=weights+time_lp-marginal[...,None]
        selected=log_type.gather(-1,request_label[:,None,None,None].expand(B,K,J,1))[...,0]
        return -torch.logsumexp(posterior+selected,-1).mean()
