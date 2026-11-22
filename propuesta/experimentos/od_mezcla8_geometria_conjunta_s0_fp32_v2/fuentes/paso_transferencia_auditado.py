"""Finite candidate selection using the actual optimizer update and a training-only audit batch.
The deterministic audit-loss check is local; it gives no guarantee on unseen data.
"""
import copy,math
import torch

def audited_transfer_step(model,optimizer,target_gradients,source_gradients,audit_training_loss,lr_scales=(1.,.5,.25,.125),tolerance=1e-10):
    params=[p for group in optimizer.param_groups for p in group['params']]
    assert len({id(p) for p in params})==len(params)
    assert len(target_gradients)==len(params) and all(len(g)==len(params) for g in source_gradients)
    for gradients in [target_gradients,*source_gradients]:
        for p,g in zip(params,gradients):
            if g is not None:assert g.shape==p.shape and torch.isfinite(g).all()
    assert lr_scales and all(0<float(s)<=1 for s in lr_scales)
    base_model=copy.deepcopy(model.state_dict());base_optimizer=copy.deepcopy(optimizer.state_dict());base_gradients=[None if p.grad is None else p.grad.detach().clone() for p in params];training_modes=[(module,module.training) for module in model.modules()];base_lrs=[g['lr'] for g in optimizer.param_groups]
    def restore(state_model,state_optimizer):
        model.load_state_dict(state_model);optimizer.load_state_dict(copy.deepcopy(state_optimizer))
    def restore_modes():
        for module,mode in training_modes:module.training=mode
    def measure(require_finite=True):
        model.eval()
        with torch.no_grad():value=float(audit_training_loss())
        if require_finite:assert math.isfinite(value),'Audit loss must be finite'
        return value
    trials=[];best_model=base_model;best_optimizer=base_optimizer;best_weights=None;best_scale=0.
    try:
        baseline=measure();repeated=measure();assert abs(baseline-repeated)<=tolerance,'Audit evaluation must be deterministic; disable evaluation noise'
        best_loss=baseline;N=len(source_gradients);weights=[tuple(0. for _ in range(N))]
        if N:weights+=[tuple(1./N for _ in range(N))]+[tuple(float(i==j) for i in range(N)) for j in range(N)]
        weights=list(dict.fromkeys(weights))
        for candidate in weights:
            gradients=[]
            for j,p in enumerate(params):
                terms=[]
                if target_gradients[j] is not None:terms.append(target_gradients[j].detach())
                for weight,g in zip(candidate,source_gradients):
                    if weight and g[j] is not None:terms.append(g[j].detach()*weight)
                gradients.append(sum(terms,torch.zeros_like(p)) if terms else None)
            for scale in lr_scales:
                restore(base_model,base_optimizer)
                for group,lr in zip(optimizer.param_groups,base_lrs):group['lr']=lr*scale
                for p,g in zip(params,gradients):p.grad=None if g is None else g.clone()
                optimizer.step();loss=measure(False);finite=math.isfinite(loss);trials.append(dict(source_weights=list(candidate),lr_scale=float(scale),audit_loss=loss if finite else None,non_increasing=finite and loss<=baseline+tolerance))
                if finite and loss<best_loss-tolerance:
                    best_loss=loss;best_weights=candidate;best_scale=float(scale);best_model=copy.deepcopy(model.state_dict());best_optimizer=copy.deepcopy(optimizer.state_dict())
        restore(best_model,best_optimizer)
        # Backtracking changes this update only, not the caller's learning-rate schedule.
        for group,lr in zip(optimizer.param_groups,base_lrs):group['lr']=lr
        final=measure();assert final<=baseline+tolerance
        for p,g in zip(params,base_gradients):p.grad=g
        restore_modes()
        return dict(accepted=best_weights is not None,status='accepted' if best_weights is not None else 'no_update',source_weights=list(best_weights) if best_weights is not None else [0.]*len(source_gradients),lr_scale=best_scale,audit_loss_before=baseline,audit_loss_after=final,trials=trials,guarantee_scope='This deterministic target-training audit batch only; no claim about validation, test or expected target risk.')
    except Exception:
        restore(base_model,base_optimizer)
        for p,g in zip(params,base_gradients):p.grad=g
        restore_modes()
        raise
