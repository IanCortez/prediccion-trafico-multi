"""Differentiable log probabilities of IID complete trajectories accepted by fixed cohort."""
import torch

def native_training_step(model,q,geo,cell,destination,heading,age,elapsed,hidden,fold=None,timestamp=None,event_history=None):
    local=model.state_representation(q,geo,cell,destination,heading)
    extra=torch.stack([age.to(local.dtype)/120,torch.full_like(age,elapsed,dtype=local.dtype)/120],-1)
    hist,hidden=model.prefix_rnn(torch.cat([local,extra],-1)[:,None],hidden)
    state=local+model.history_projection(hist[:,0])
    if hasattr(model,'action_with_history'):
        if fold is None:
            assert timestamp is not None
            prior=model.historical_probability(cell,destination,heading,timestamp,elapsed,age)
        else:
            assert event_history is not None,'Never use full Train event history for a Train query'
            from historia_eventos import event_groups
            cc=cell.detach().cpu().numpy();dd=destination.detach().cpu().numpy()
            hh=heading.detach().cpu().numpy();aa=age.detach().cpu().numpy()
            tt=timestamp.detach().cpu().numpy();ff=fold.detach().cpu().numpy()
            group=event_groups(cc,hh,dd,tt,elapsed,aa)
            prior=torch.tensor(event_history.lookup(group,aa,ff),device=q.device)
        motion,stop=model.action_with_history(state,geo,cell,destination,heading,prior,fold)
    else:
        if fold is not None:assert model.counts_fold.numel()>0,'Fold buffers required during policy training'
        motion,stop=model.action_probabilities(state,geo,cell,destination,heading,fold)
    return motion,stop,hidden

def sample_cohort_trajectories(model,x,origin,destination,particles=8,candidates=16,max_rounds=4,horizon=120,min_steps=4,fold=None,timestamp=None,event_history=None,step_function=None):
    # Labels are deliberately absent from this API. Dropout shared by particles
    # would violate the IID marginal-law assumption of the CRPS estimator.
    assert particles>=3 and candidates>=particles and max_rounds>=1
    assert 0<=min_steps<horizon<=120
    for module in model.modules():
        assert not (isinstance(module,torch.nn.Dropout) and module.training),'Disable encoder dropout for policy sampling'
        assert not (isinstance(module,torch.nn.GRU) and module.training and module.dropout>0)
    step=native_training_step if step_function is None else step_function
    B=len(x);C=candidates
    q,*_=model.encode_od(x,origin,destination)
    def expand(v):return v[:,None].expand(B,C,*v.shape[1:]).reshape(B*C,*v.shape[1:])
    qq=expand(q);geo=expand(x[:,:4]);dest=expand(destination)
    ff=None if fold is None else expand(fold)
    ts=None if timestamp is None else expand(timestamp)
    accepted_times=[[] for _ in range(B)]
    accepted_logp=[[] for _ in range(B)]
    accepted_counts=[0]*B
    attempted=0;zero_motion=torch.zeros((),device=x.device,dtype=torch.long)
    for round_index in range(max_rounds):
        cell=expand(origin);heading=torch.full_like(cell,4);age=torch.zeros_like(cell)
        hidden=qq[None].contiguous()
        live=torch.ones(B*C,device=x.device,dtype=torch.bool)
        ended=torch.full_like(cell,-1)
        logp=torch.zeros(B*C,device=x.device,dtype=q.dtype)
        for t in range(horizon):
            motion,stop,hidden=step(model,qq,geo,cell,dest,heading,age,t,hidden,ff,ts,event_history)
            died=live&(torch.rand(stop.shape,device=stop.device)<stop.detach())
            # Safe logs at structural probabilities0or1 avoid inactive-branch NaNs.
            log_stop=torch.where(stop>0,stop,torch.ones_like(stop)).log()
            log_continue=torch.where(stop<1,1-stop,torch.ones_like(stop)).log()
            logp=logp+live.to(q.dtype)*torch.where(died,log_stop,log_continue)
            ended=torch.where(died,torch.full_like(ended,t),ended)
            live=live&(~died)
            if t==horizon-1:break
            action=torch.multinomial(motion.detach(),1)[:,0]
            probability=motion.gather(1,action[:,None])[:,0]
            zero_motion=zero_motion+((probability<=0)&live).sum()
            log_motion=torch.where(probability>0,probability,torch.ones_like(probability)).log()
            logp=logp+live.to(q.dtype)*log_motion
            move=(action>0)&live
            a=(action-1).clamp_min(0)
            nxt=model.neighbors[cell,a]
            cell=torch.where(move,nxt,cell)
            heading=torch.where(move,model.outgoing_heading[a],heading)
            age=torch.where(move,torch.zeros_like(age),torch.where(live,age+1,age))
        ended=ended.reshape(B,C);logp=logp.reshape(B,C)
        accepted=(ended>=min_steps)&(ended<horizon)
        attempted+=C
        for b in range(B):
            need=particles-accepted_counts[b]
            if need==0:continue
            index=torch.nonzero(accepted[b],as_tuple=False).flatten()[:need]
            accepted_times[b].append(ended[b,index])
            accepted_logp[b].append(logp[b,index])
            accepted_counts[b]+=len(index)
        if min(accepted_counts)==particles:break
    assert int(zero_motion)==0,'A zero-probability motion was sampled'
    complete=torch.tensor([n==particles for n in accepted_counts],device=x.device)
    times=[];log_probability=[]
    for b in range(B):
        if accepted_counts[b]==particles:
            times.append(torch.cat(accepted_times[b]).to(q.dtype)*15)
            log_probability.append(torch.cat(accepted_logp[b]))
        else:
            times.append(torch.zeros(particles,device=x.device,dtype=q.dtype))
            log_probability.append(torch.zeros(particles,device=x.device,dtype=q.dtype))
    times=torch.stack(times);log_probability=torch.stack(log_probability)
    assert torch.isfinite(log_probability).all()
    return dict(times=times,log_probability=log_probability,complete=complete,accepted_counts=accepted_counts,candidate_attempts_per_query=attempted,rounds=round_index+1,cohort_min_steps=min_steps,horizon=horizon)
