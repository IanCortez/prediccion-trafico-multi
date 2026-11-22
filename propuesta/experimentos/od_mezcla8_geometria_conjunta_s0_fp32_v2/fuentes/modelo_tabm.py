import torch
from torch import nn
import tabm
import rtdl_num_embeddings as rtdl
class CityTabM(nn.Module):
    def __init__(self,mean,std,keep,bins,center,k=16,width=512):
        super().__init__()
        self.register_buffer("mean",torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer("std",torch.as_tensor(std,dtype=torch.float32))
        self.register_buffer("keep",torch.as_tensor(keep,dtype=torch.long))
        self.register_buffer("center",torch.tensor(float(center)))
        embeddings=rtdl.PiecewiseLinearEmbeddings(bins,d_embedding=16,activation=False,version="B")
        self.backbone=tabm.TabM.make(n_num_features=len(keep),num_embeddings=embeddings,d_out=1,k=k,d_block=width,n_blocks=2,dropout=.1,arch_type="tabm")
    def forward(self,x):
        z=(x[:,self.keep]-self.mean)/self.std
        return self.center+600*self.backbone(z).squeeze(-1)

