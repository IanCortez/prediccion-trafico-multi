"""Exact finite-horizon semi-Markov first-arrival kernel. No observed query path."""
import torch

def arrival_distribution(policy,move_wait,end_wait,end_probability,neighbors,origin,destination):
 # policy[B,N,J], move_wait[B,N,H-1] on durations1..H-1,
 # end_wait[B,H] on durations0..H-1. Movement always consumes positive time.
 B,N,J=policy.shape;H=end_wait.shape[-1];assert move_wait.shape==(B,N,H-1);assert neighbors.shape==(N,J);assert int(neighbors.max())<N
 device=policy.device;dtype=policy.dtype;idx=neighbors.clamp_min(0).reshape(1,-1).expand(B,-1);policy=policy*(neighbors>=0)[None,:].to(dtype);leave=torch.ones((B,N),device=device,dtype=dtype).scatter(1,destination[:,None],1-end_probability[:,None]);initial=torch.zeros((B,N),device=device,dtype=dtype).scatter(1,origin[:,None],1.)
 if torch.is_grad_enabled():
  history=[initial]
  for t in range(1,H):
   past=torch.stack(history[::-1],-1);depart=(past*move_wait[:,:,:t]).sum(-1)*leave;flow=(depart[:,:,None]*policy).reshape(B,-1);current=torch.zeros_like(initial).scatter_add(1,idx,flow);history.append(current)
  arrivals=torch.stack([a.gather(1,destination[:,None])[:,0] for a in history],-1)
 else:
  reverse=torch.zeros((B,N,H),device=device,dtype=dtype);reverse[:,:,-1]=initial
  for t in range(1,H):
   depart=(reverse[:,:,H-t:]*move_wait[:,:,:t]).sum(-1)*leave;flow=(depart[:,:,None]*policy).reshape(B,-1);reverse[:,:,H-1-t]=torch.zeros_like(initial).scatter_add(1,idx,flow)
  arrivals=reverse.gather(1,destination[:,None,None].expand(B,1,H))[:,0].flip(-1)
 n_fft=1<<((2*H-2).bit_length());p=torch.fft.irfft(torch.fft.rfft(arrivals,n=n_fft)*torch.fft.rfft(end_wait,n=n_fft),n=n_fft)[:,:H]*end_probability[:,None]
 return p.clamp_min(0)
