"""Prefix-memory road/time model. Public rollout accepts no teacher targets."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class ProgressNet(nn.Module):
    def __init__(self,mean,std,graph):
        super().__init__();self.nroads=len(graph['static']);self.styles=4
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32))
        for key in ['static','chain_length','chain_src','chain_dst','neighbors','neighbor_valid']:
            dtype=torch.float32 if key in ['static','chain_length'] else torch.bool if key=='neighbor_valid' else torch.long
            value=torch.as_tensor(graph[key],dtype=dtype)
            pad=torch.zeros((1,)+value.shape[1:],dtype=dtype)
            if key=='neighbors':pad.fill_(self.nroads)
            self.register_buffer(key,torch.cat([value,pad]),persistent=False)
        # Reconstructible two-GiB public matrix NEVER in model checkpoints.
        self.register_buffer('D',torch.as_tensor(graph['D'].copy()),persistent=False)
        for key in ['summer_start','summer_end']:
            self.register_buffer(key,torch.as_tensor(graph[key],dtype=torch.float64),persistent=False)
        self.embedding=nn.Embedding(self.nroads+1,32,padding_idx=self.nroads)
        nn.init.normal_(self.embedding.weight,std=.04)
        with torch.no_grad():self.embedding.weight[self.nroads].zero_()
        self.road_encoder=nn.Sequential(nn.Linear(76,64),nn.SiLU(),nn.LayerNorm(64))
        self.encoder=nn.Sequential(nn.Linear(len(mean),384),nn.SiLU(),nn.LayerNorm(384),
            nn.Linear(384,256),nn.SiLU(),nn.LayerNorm(256))
        self.prior=nn.Linear(256,4)
        self.recognition=nn.Sequential(nn.Linear(274,128),nn.SiLU(),nn.Linear(128,4))
        self.style_embedding=nn.Parameter(torch.randn(4,16)*.04)
        self.root_head=nn.Sequential(nn.Linear(256,128),nn.SiLU(),nn.Linear(128,8))
        self.attach_head=nn.Sequential(nn.Linear(400,96),nn.SiLU(),nn.Linear(96,1))
        self.state_input=nn.Sequential(nn.Linear(414,128),nn.SiLU())
        self.init_hidden=nn.Sequential(nn.Linear(336,128),nn.Tanh())
        self.gru=nn.GRU(128,128,batch_first=True)
        self.time_head=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,3))
        self.policy=nn.Sequential(nn.Linear(196,96),nn.SiLU(),nn.Linear(96,1))
        self.eos=nn.Linear(128,1)
        grid=torch.arange(60,1800,15,dtype=torch.float32)
        self.register_buffer('grid',grid);self.register_buffer('log_edges',(grid[:-1]+7.5).log())
        nn.init.normal_(self.root_head[-1].weight,std=.005)
        nn.init.zeros_(self.root_head[-1].bias)

    def root(self,x):
        return self.encoder(((x-self.mean)/self.std).clamp(-30,30))
    def roads(self,ids):
        nei=self.neighbors[ids];v=self.neighbor_valid[ids]
        pooled=(self.embedding(nei)*v[...,None]).sum(-2)/v.sum(-1,keepdim=True).clamp_min(1)
        return self.road_encoder(torch.cat([self.embedding(ids),pooled,self.static[ids]],-1))
    def root_logprob(self,h):
        raw=self.root_head(h).reshape(-1,4,2)
        loc=math.log(600)+2.5*torch.tanh(raw[...,0]/2.5)
        sigma=(.12+F.softplus(raw[...,1]-1.5)).clamp_max(1.5)
        cdf=torch.special.ndtr((self.log_edges-loc[...,None])/sigma[...,None])
        p=torch.cat([cdf[...,:1],cdf[...,1:]-cdf[...,:-1],1-cdf[...,-1:]],-1).clamp_min(1e-7)
        p=p/p.sum(-1,keepdim=True)
        return torch.logsumexp(p.log()+self.prior(h).log_softmax(-1)[...,None],1)[:,None]
    def root_quantiles(self,h):
        cdf=self.root_logprob(h).exp()[:,0].cumsum(-1)
        ind=(cdf[...,None]>=torch.tensor([.1,.5,.9],device=h.device)).float().argmax(1)
        return self.grid[ind]

    def remaining_distance(self,cur,entry,dest,dpos):
        remain=self.chain_length[cur]-entry+self.D[self.chain_dst[cur],self.chain_src[dest]]+dpos
        direct=(cur==dest)&(dpos>=entry-1e-4)
        remain=torch.where(direct,torch.minimum(remain,(dpos-entry).clamp_min(0)),remain)
        valid=torch.isfinite(remain)&(remain>=-.1)
        return torch.where(valid,remain.clamp(0,200000),torch.full_like(remain,200000)),valid
    def calendar(self,unix):
        assert unix.dtype==torch.float64
        summer=((unix[...,None]>=self.summer_start)&(unix[...,None]<self.summer_end)).any(-1)
        local=unix+summer.to(torch.float64)*3600
        day=local.remainder(86400)/86400
        week=(torch.floor(local/86400)+3).remainder(7)/7
        hour=local.remainder(3600)/3600
        return torch.stack([torch.sin(2*math.pi*day),torch.cos(2*math.pi*day),
            torch.sin(2*math.pi*week),torch.cos(2*math.pi*week),
            torch.sin(2*math.pi*hour),torch.cos(2*math.pi*hour)],-1).float()
    def state_vectors(self,root,road,destroad,style,cur,entry,dest,dpos,unix):
        remain,valid=self.remaining_distance(cur,entry,dest,dpos)
        clen=self.chain_length[cur].clamp_min(.001);dlen=self.chain_length[dest].clamp_min(.001)
        meta=torch.stack([entry/clen,dpos/dlen,torch.log1p(remain)/10,remain/10000,
            (cur==dest).float(),valid.float(),torch.log1p(clen)/10,torch.log1p(dlen)/10],-1)
        vectors=torch.cat([root,road,destroad,style,self.calendar(unix),meta],-1)
        return self.state_input(vectors),remain,valid
    def initial(self,root,destroad,style):
        return self.init_hidden(torch.cat([root,destroad,style],-1))
    def outputs(self,h,cur,entry,dest,dpos,road_table):
        raw=self.time_head(h)
        # Pace positive; residual delay is statistical, not an annotated stop.
        pace=.025+F.softplus(raw[...,0])/5
        delay=F.softplus(raw[...,1])*5
        remain,reachable=self.remaining_distance(cur,entry,dest,dpos)
        value=(remain/8+15)*torch.exp(2*torch.tanh(raw[...,2]/2))
        nei=self.neighbors[cur];v=self.neighbor_valid[cur]
        de=dest[...,None].expand_as(nei);dp=dpos[...,None].expand_as(nei)
        nr,nv=self.remaining_distance(nei,torch.zeros_like(nei,dtype=torch.float32),de,dp)
        turn=torch.stack([torch.log1p(nr)/10,nr/10000,nv.float(),
            (nei==de).float()],-1)
        hh=h[...,None,:].expand(h.shape[:-1]+(nei.shape[-1],h.shape[-1]))
        logits=self.policy(torch.cat([hh,road_table[nei],turn],-1))[...,0].masked_fill(~v,-1e9)
        can_end=(cur==dest)&(dpos>=entry-1e-4)
        eos=self.eos(h)[...,0].masked_fill(~can_end,-1e9)
        return pace,delay,value,torch.cat([logits,eos[...,None]],-1),reachable

    def attachment_logprob(self,h,chain,fraction,error_m,remaining_km,reachable,road_table):
        B=len(h);oc=chain[:,0,:,None].expand(-1,-1,4).reshape(B,16).long()
        dc=chain[:,1,None,:].expand(-1,4,-1).reshape(B,16).long()
        of=fraction[:,0,:,None].expand(-1,-1,4).reshape(B,16).float()
        df=fraction[:,1,None,:].expand(-1,4,-1).reshape(B,16).float()
        oe=error_m[:,0,:,None].expand(-1,-1,4).reshape(B,16).float()
        de=error_m[:,1,None,:].expand(-1,4,-1).reshape(B,16).float()
        r=remaining_km.reshape(B,16).float()
        meta=torch.stack([of,df,oe/100,de/100,torch.log1p(r),r/10,
            torch.log1p(self.chain_length[oc])/10,torch.log1p(self.chain_length[dc])/10,
            (oc==dc).float(),reachable.reshape(B,16).float(),
            oe.clamp_max(1000)/1000,de.clamp_max(1000)/1000,
            of*df,of-df,torch.sqrt(r.clamp_min(0)),torch.ones_like(r)],-1)
        ctx=torch.cat([h[:,None].expand(-1,16,-1),road_table[oc],road_table[dc],meta],-1)
        score=self.attach_head(ctx)[...,0]-.5*((oe/30).square()+(de/30).square())
        valid=reachable.reshape(B,16).bool()
        safe=valid.clone();safe[:,0]|=~valid.any(1)
        return score.masked_fill(~safe,-1e9).log_softmax(-1)

    def teacher_forward(self,x,private,noise):
        h=self.root(x);b=private;B,L=b['chain'].shape;cur=b['chain']
        q=self.recognition(torch.cat([h,b['recognition']],-1)).softmax(-1)
        # One parent-specific Gumbel draw per epoch, reused across its fragments.
        soft=((q.clamp_min(1e-8).log()+noise)/.7).softmax(-1)
        one=F.one_hot(soft.argmax(-1),4).float();weights=one+soft-soft.detach()
        style=weights@self.style_embedding
        table=self.roads(torch.arange(self.nroads+1,device=x.device));dest=b['destination']
        root=h[:,None].expand(-1,L,-1);dr=table[dest][:,None].expand(-1,L,-1)
        dests=dest[:,None].expand(-1,L);dpos=b['destination_m'][:,None].expand(-1,L)
        unix=b['timestamp'][:,None].double()+b['elapsed_s'].double()
        inputs,_,_=self.state_vectors(root,table[cur],dr,style[:,None].expand(-1,L,-1),
            cur,b['entry_m'],dests,dpos,unix)
        initial=self.initial(h,table[dest],style)[None]
        # Only corrected contiguous fragments, every valid token supervised.
        out,_=self.gru(inputs,initial)
        pace,delay,value,logits,_=self.outputs(out,cur,b['entry_m'],dests,dpos,table)
        valid=b['valid'];dt=pace*b['distance_m']+delay
        time_loss=F.smooth_l1_loss(torch.log1p(dt[valid]),torch.log1p(b['local_s'][valid]))
        value_loss=F.smooth_l1_loss(torch.log1p(value[valid]),torch.log1p(b['remaining_s'][valid]))
        rv=valid&(b['route_target']>=0)
        route_loss=F.cross_entropy(logits[rv],b['route_target'][rv]) if rv.any() else logits.sum()*0
        prior=self.prior(h).log_softmax(-1)
        kl=(q*(q.clamp_min(1e-8).log()-prior)).sum(-1).mean()
        return .2*route_loss+.2*time_loss+.1*value_loss+.02*kl,dict(
            route=route_loss.detach(),local=time_loss.detach(),value=value_loss.detach(),kl=kl.detach(),
            style_mass=q.detach().mean(0),route_accuracy=(logits[rv].argmax(-1)==b['route_target'][rv]).float().mean().detach())

    @torch.no_grad()
    def public_rollout(self,x,timestamp,chain,fraction,error_m,remaining_km,reachable,max_steps=192,top_pairs=4):
        """Public-only; no route, target length, observed elapsed, driver or T arguments."""
        assert not self.training and max_steps>0
        B=len(x);h=self.root(x);table=self.roads(torch.arange(self.nroads+1,device=x.device))
        alp=self.attachment_logprob(h,chain,fraction,error_m,remaining_km,reachable,table)
        pair=alp.topk(top_pairs,-1).indices
        probs=alp.exp().gather(1,pair);pairvalid=reachable.reshape(B,16).gather(1,pair).bool()
        prior=self.prior(h).softmax(-1)
        weights=(probs[:,:,None]*prior[:,None]).reshape(B,-1)
        weights=weights*pairvalid[:,:,None].expand(-1,-1,4).reshape(B,-1)
        oc=(pair//4);dc=pair%4
        def repeat_style(t):return t[:,:,None].expand(-1,-1,4).reshape(-1)
        cur=repeat_style(chain[:,0].long().gather(1,oc))
        dest=repeat_style(chain[:,1].long().gather(1,dc))
        entry=repeat_style(fraction[:,0].float().gather(1,oc))*self.chain_length[cur]
        dpos=repeat_style(fraction[:,1].float().gather(1,dc))*self.chain_length[dest]
        root=h[:,None,None].expand(-1,top_pairs,4,-1).reshape(-1,256)
        style=self.style_embedding[None,None].expand(B,top_pairs,-1,-1).reshape(-1,16)
        dr=table[dest];hidden=self.initial(root,dr,style)
        unix=timestamp[:,None,None].double().expand(-1,top_pairs,4).reshape(-1)
        total=torch.zeros(len(cur),device=x.device,dtype=torch.float64)
        active=repeat_style(pairvalid);done=torch.zeros_like(active)
        steps=torch.zeros(len(cur),device=x.device);tail=torch.zeros_like(steps)
        for step in range(max_steps):
            inp,_,_=self.state_vectors(root,table[cur],dr,style,cur,entry,dest,dpos,unix+total)
            out,hn=self.gru(inp[:,None],hidden[None]);newhidden=hn[0]
            hidden=torch.where(active[:,None],newhidden,hidden)
            pace,delay,value,logits,reach=self.outputs(hidden,cur,entry,dest,dpos,table)
            choice=logits.argmax(-1);can_end=(cur==dest)&(dpos>=entry-1e-4)
            eos=active&(choice==6)&can_end
            dead=active&(~reach|(~self.neighbor_valid[cur].any(-1)&~can_end))
            normal=active&~dead
            distance=torch.where(eos,(dpos-entry).clamp_min(0),(self.chain_length[cur]-entry).clamp_min(0))
            dt=pace*distance+delay
            total=total+torch.where(normal,dt,torch.where(dead,value,torch.zeros_like(value))).double()
            steps+=normal.float();tail+=dead.float()
            done|=eos;moving=normal&~eos
            nxt=self.neighbors[cur].gather(1,choice.clamp_max(5)[:,None])[:,0]
            cur=torch.where(moving,nxt,cur);entry=torch.where(moving,torch.zeros_like(entry),entry)
            active=moving
            if not active.any():break
        if active.any():
            inp,_,_=self.state_vectors(root,table[cur],dr,style,cur,entry,dest,dpos,unix+total)
            _,hn=self.gru(inp[:,None],hidden[None])
            _,_,value,_,_=self.outputs(hn[0],cur,entry,dest,dpos,table)
            total+=torch.where(active,value,torch.zeros_like(value)).double()
            tail+=active.float()
        eta=total.float().reshape(B,-1).clamp(0,20000)
        st=steps.reshape(B,-1);ta=tail.reshape(B,-1);dn=done.float().reshape(B,-1)
        rootq=self.root_quantiles(h);mass=weights.sum(-1)
        fallback=(1-mass).clamp(0,1);mean=(eta*weights).sum(-1)+fallback*rootq[:,1]
        second=(eta.square()*weights).sum(-1)+fallback*rootq[:,1].square()
        var=(second-mean.square()).clamp_min(0).sqrt()
        # Fixed 30 public summaries, no full query paths retained.
        pathq=[]
        order=eta.argsort(-1);sortedeta=eta.gather(1,order);sw=weights.gather(1,order)
        # Include omitted attachment mass as root fallback before quantiles.
        all_eta=torch.cat([eta,rootq[:,1:2]],1);all_w=torch.cat([weights,fallback[:,None]],1)
        order=all_eta.argsort(-1);se=all_eta.gather(1,order);cw=all_w.gather(1,order).cumsum(-1)
        for q in [.1,.25,.5,.75,.9]:
            idx=(cw>=q-1e-7).float().argmax(-1);pathq.append(se.gather(1,idx[:,None])[:,0])
        style_eta=(eta.reshape(B,top_pairs,4)*probs[:,:,None]*pairvalid[:,:,None]).sum(1)
        style_eta=style_eta+(1-(probs*pairvalid).sum(1))[:,None]*rootq[:,1:2]
        summary=torch.cat([rootq,torch.stack([mean,var,fallback,(st*weights).sum(-1),
            (ta*weights).sum(-1),(dn*weights).sum(-1),-(alp.exp()*alp).sum(-1),
            prior.max(-1).values,-(prior*prior.clamp_min(1e-8).log()).sum(-1)],1),
            torch.stack(pathq,1),style_eta,F.pad(probs,(0,4-top_pairs)),prior,remaining_km.reshape(B,16).min(-1).values[:,None]],1)
        assert summary.shape==(B,30) and torch.isfinite(summary).all()
        return summary
