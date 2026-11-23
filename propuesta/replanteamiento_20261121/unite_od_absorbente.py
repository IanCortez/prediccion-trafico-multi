"""Target-free street-segment OD controller; arrivals receive terminal time exactly once."""
import torch,torch.nn.functional as F
@torch.no_grad()
def absorbing_od_forward(self,o_seg,d_seg,depart,day,discrete_depart,max_steps=50):
 assert max_steps>=1
 o=o_seg+1;d=d_seg+1;B=len(o);device=o.device
 odt=self.odt_encoder(o,d,depart,discrete_depart);t_emb=self.emb_departure(depart)
 slots=day*self.num_slots+depart;des_emb=self.get_seg_emb(d,slots,t_emb)
 history=torch.zeros((max_steps+1,B,self.hid_dim),device=device);history[0]=des_emb
 hidden=odt[None].repeat(self.num_layers,1,1);current=o
 rows=torch.arange(B,device=device);visits=torch.zeros((B,self.seg_size+1),dtype=torch.long,device=device);visits[rows,current]=1
 times=torch.zeros((max_steps,B),device=device);ids=torch.full((max_steps,B),-1,dtype=torch.long,device=device)
 lengths=torch.full((B,),max_steps,dtype=torch.long,device=device);done=torch.zeros(B,dtype=torch.bool,device=device);arrived=torch.zeros_like(done)
 des_cols,idx,t_idx,d_cords,t_graph=self.get_dest_info(d,slots)
 for step in range(max_steps):
  active=~done;embedding=self.get_seg_emb(current,slots,t_emb);history[step+1]=embedding
  prediction,next_hidden=self.prog_state_encoding(current,embedding,hidden,history,step+1)
  hidden=torch.where(active[None,:,None],next_hidden,hidden)
  eta,eta_embedding=self.eta_predictor(odt,prediction,current,slots,t_idx,t_graph)
  times[step]=torch.where(active,eta,torch.zeros_like(eta));ids[step]=torch.where(active,current-1,torch.full_like(current,-1))
  reached=active&(current==d);arrived|=reached;lengths=torch.where(reached,torch.full_like(lengths,step+1),lengths);done|=reached
  if done.all() or step+1==max_steps:break
  candidate,nbrs,mask,_,_=self.candi_encoder(current,des_cols,idx,d_cords,t_idx,slots)
  prediction=self.eta_fuse_mlp(torch.cat([prediction,eta_embedding],-1))
  inputs=torch.cat([candidate,prediction[:,None].expand(-1,candidate.shape[1],-1),des_emb[:,None].expand(-1,candidate.shape[1],-1)],-1)
  scores=self.fc_id_out(inputs).squeeze(-1).masked_fill(mask==0,-1e6);rank=scores.argsort(-1,descending=True)
  chosen=nbrs[rows,rank[:,0]]
  for k in range(1,self.candi_encoder.max_deg):
   acceptable=visits[rows,chosen]<self.freq_limit
   chosen=torch.where(acceptable,chosen,nbrs[rows,rank[:,k]])
  destination_available=((nbrs==d[:,None])&(mask>0)).any(-1);chosen=torch.where(destination_available,d,chosen)
  dead=(mask.sum(-1)==0)&(~done);lengths=torch.where(dead,torch.full_like(lengths,step+1),lengths);done|=dead
  advance=~done;current=torch.where(advance,chosen,current);visits[rows,current]+=advance.long()
 return times,ids,lengths,~arrived
