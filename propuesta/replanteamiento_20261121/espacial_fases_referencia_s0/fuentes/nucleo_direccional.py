"""Exact finite-horizon semi-Markov absorption at multiple directional destination states."""
import torch
def directional_arrival(policy,move_wait,end_wait,end_probability,neighbors,origin,end_states):
 B,S,J=policy.shape;H=end_wait.shape[-1];K=end_states.shape[1];assert move_wait.shape==(B,S,H-1) and end_wait.shape==(B,K,H) and end_probability.shape==(B,K);assert neighbors.shape in [(S,J),(B,S,J)]
 nb=neighbors[None].expand(B,-1,-1) if neighbors.ndim==2 else neighbors
 assert int(nb.max())<S and int(end_states.min())>=0 and int(end_states.max())<S
 dtype=policy.dtype;device=policy.device;idx=nb.clamp_min(0).reshape(B,-1);policy=policy*(nb>=0).to(dtype);leave=torch.ones((B,S),device=device,dtype=dtype).scatter(1,end_states,1-end_probability);initial=torch.zeros_like(leave).scatter(1,origin[:,None],1.)
 if torch.is_grad_enabled():
  history=[initial]
  for t in range(1,H):
   depart=(torch.stack(history[::-1],-1)*move_wait[:,:,:t]).sum(-1)*leave;flow=(depart[:,:,None]*policy).reshape(B,-1);history.append(torch.zeros_like(initial).scatter_add(1,idx,flow))
  arrivals=torch.stack([h.gather(1,end_states) for h in history],-1)
 else:
  reverse=torch.zeros((B,S,H),device=device,dtype=dtype);reverse[:,:,-1]=initial
  for t in range(1,H):
   depart=(reverse[:,:,H-t:]*move_wait[:,:,:t]).sum(-1)*leave;flow=(depart[:,:,None]*policy).reshape(B,-1);reverse[:,:,H-1-t]=torch.zeros_like(initial).scatter_add(1,idx,flow)
  arrivals=reverse.gather(1,end_states[:,:,None].expand(B,K,H)).flip(-1)
 n_fft=1<<((2*H-2).bit_length());p=torch.fft.irfft(torch.fft.rfft(arrivals,n=n_fft)*torch.fft.rfft(end_wait,n=n_fft),n=n_fft)[:,:,:H]*end_probability[:,:,None]
 return p.sum(1).clamp_min(0)

