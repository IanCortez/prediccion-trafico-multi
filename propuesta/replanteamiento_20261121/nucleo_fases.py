"""Finite-horizon arrival with mixtures of shifted geometric residence distributions.

Each component k spends at least lag[k]+1 time steps in a spatial state.
After that delay, its remaining residence is geometric with hazard[k].
The recurrence is equivalent to explicit semi-Markov convolution, while
keeping only K reservoirs and the most recent max(lag)+1 arrival vectors.
"""
import torch

def geometric_wait_pmf(weights,hazards,lags,horizon):
    n=torch.arange(1,horizon,device=weights.device,dtype=weights.dtype)
    lag_tensor=torch.as_tensor(lags,device=weights.device,dtype=weights.dtype)
    exponent=n[None,:]-lag_tensor[:,None]-1
    component=hazards[...,None]*torch.pow(1-hazards[...,None],exponent.clamp_min(0))
    component=component*(exponent>=0)
    return (weights[...,None]*component).sum(-2)

def phase_arrival(policy,weights,hazards,lags,end_wait,end_probability,neighbors,origin,end_states):
    B,S,J=policy.shape
    K=weights.shape[-1]
    H=end_wait.shape[-1]
    assert weights.shape==hazards.shape==(B,S,K)
    lag_values=lags.tolist() if isinstance(lags,torch.Tensor) else list(lags)
    assert len(lag_values)==K and min(lag_values)>=0
    assert end_states.shape==end_probability.shape
    nb=neighbors[None].expand(B,-1,-1) if neighbors.ndim==2 else neighbors
    assert nb.shape==(B,S,J)
    index=nb.clamp_min(0).reshape(B,-1)
    policy=policy*(nb>=0).to(policy.dtype)
    initial=torch.zeros((B,S),device=policy.device,dtype=policy.dtype).scatter(1,origin[:,None],1.)
    leave=torch.ones_like(initial).scatter(1,end_states,1-end_probability)
    reservoir=torch.zeros_like(weights)
    zero=torch.zeros_like(initial)
    arrivals=[initial]
    terminal_arrivals=[initial.gather(1,end_states)]
    maxlag=max(lag_values)
    for t in range(1,H):
        delayed=torch.stack([arrivals[-1-int(lag)] if len(arrivals)>int(lag) else zero for lag in lag_values],-1)
        reservoir=(1-hazards)*reservoir+weights*delayed
        depart=(hazards*reservoir).sum(-1)*leave
        flow=(depart[:,:,None]*policy).reshape(B,-1)
        current=torch.zeros_like(initial).scatter_add(1,index,flow)
        arrivals.append(current)
        if len(arrivals)>maxlag+1:arrivals.pop(0)
        terminal_arrivals.append(current.gather(1,end_states))
    at_end=torch.stack(terminal_arrivals,-1)
    n_fft=1<<((2*H-2).bit_length())
    final=torch.fft.irfft(torch.fft.rfft(at_end,n=n_fft)*torch.fft.rfft(end_wait,n=n_fft),n=n_fft)[...,:H]
    return (final*end_probability[:,:,None]).sum(1).clamp_min(0)

