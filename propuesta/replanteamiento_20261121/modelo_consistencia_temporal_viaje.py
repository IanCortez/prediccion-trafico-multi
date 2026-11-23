"""Whole-trip time consistency correction; frozen prefix model, no query target fields."""
import torch
from torch import nn
from torch.nn import functional as F
from modelo_progreso_recurrente import ProgressNet

class ConsistentProgressNet(ProgressNet):
    def __init__(self,mean,std,graph):
        super().__init__(mean,std,graph)
        for p in self.parameters():p.requires_grad_(False)
        self.time_correction=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,2))
        nn.init.zeros_(self.time_correction[-1].weight);nn.init.zeros_(self.time_correction[-1].bias)

    def corrected_local(self,hidden,pace,delay):
        bounded=torch.tanh(self.time_correction(hidden))
        factors=bounded.exp()
        return pace*factors[...,0],delay*factors[...,1],bounded

    def outputs(self,h,cur,entry,dest,dpos,road_table):
        pace,delay,value,logits,reachable=super().outputs(h,cur,entry,dest,dpos,road_table)
        pace,delay,_=self.corrected_local(h,pace,delay)
        return pace,delay,value,logits,reachable

    def private_local_prediction(self,x,teacher,road_table):
        """PRIVATE Train prefix/clock/positions. No targetT/recognition posterior used."""
        b=teacher;B,L=b['chain'].shape;cur=b['chain']
        with torch.no_grad():
            root=self.root(x);dest=b['destination']
            style=self.style_embedding[self.prior(root).argmax(-1)]
            dr=road_table[dest]
            inputs,_,_=self.state_vectors(root[:,None].expand(-1,L,-1),road_table[cur],
                dr[:,None].expand(-1,L,-1),style[:,None].expand(-1,L,-1),
                cur,b['entry_m'],dest[:,None].expand(-1,L),
                b['destination_m'][:,None].expand(-1,L),
                b['timestamp'][:,None].double()+b['elapsed_s'].double())
            initial=self.initial(root,dr,style)[None]
            hidden,_=self.gru(inputs,initial)
            raw=self.time_head(hidden)
            pace=.025+F.softplus(raw[...,0])/5
            delay=F.softplus(raw[...,1])*5
        pace,delay,bounded=self.corrected_local(hidden,pace,delay)
        return pace*b['distance_m']+delay,bounded
