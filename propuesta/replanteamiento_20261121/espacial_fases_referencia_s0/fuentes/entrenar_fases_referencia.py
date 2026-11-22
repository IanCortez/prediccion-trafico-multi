"""Train a physical correction of the protected OD distribution, using compact residence dynamics."""
from base_replanteo import *
from modelo_dir_fases import PhaseDirectionalArrival
from modelo_direccional import heading_ids
from modelo_mezcla_rutas import make_duration_model
from inferencia import feature_input
import torch,argparse,random,shutil

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--id',required=True)
    ap.add_argument('--steps',type=int,default=6000)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    access=deny_test()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.cuda.set_per_process_memory_fraction(.40,0)
    torch.set_float32_matmul_precision('high')
    run=OUT/args.id
    assert run.resolve().parent==OUT
    assert args.resume or not run.exists()
    assert (OUT/'espacial_prior_cf_s0/resultado.json').exists()
    assert (OUT/'cabeza_fases_cpu/resultado.json').exists()
    run.mkdir(exist_ok=args.resume)
    (run/'fuentes').mkdir(exist_ok=args.resume)
    start=time.time()
    cfg=dict(id=args.id,process_id=os.getpid(),seed=0,steps=args.steps,microbatch=8,accumulation=2,effective_batch=16,auxiliary_batch=64,auxiliary_weight=.1,structural_total_weight=.5,lr=.0001,selection_n=4096,selection_interval=500,selection_patience=6,minimum_steps=3000,inference_batch=32,reference_canonical_batch=2048,inference_precision='FP32 highest',training_precision='FP32 high',GPU_memory_fraction=.4,max_GPU_jobs=1,max_training_seconds=5400,query='Same216 OD/departure/static/history features; no true query path/time/count',reference='Protected best116-bin distribution, frozen; Train and Validation probabilities cached only in RAM',residence='8 shifted-geometric components; exact reservoir recurrence; total60..1785 conditioning only after spatial absorption',prior_exclusion='All current Train query weekfold counts subtracted; Validation uses full Train counts',objective='Mixed NLL+5RPS +.5*(structural NLL+5RPS) +.1*Train local dynamics likelihood',Test_access=False)
    if args.resume:
        prior_config=read(run/'protocolo.json')
        assert prior_config['steps']==args.steps
        cfg=prior_config
        assert sha(__file__)==sha(run/'fuentes/entrenar_fases_referencia.py')
        write(run/'sesion_reanudada.json',dict(pid=os.getpid(),started_unix=time.time()))
    else:write(run/'protocolo.json',cfg)
    for name in ['entrenar_fases_referencia.py','modelo_dir_fases.py','nucleo_fases.py','modelo_dir_prior.py','modelo_direccional.py','modelo_espacial.py','nucleo_direccional.py','nucleo_semi_markov.py','base_replanteo.py']:
        if not args.resume:shutil.copy2(OUT/name,run/'fuentes'/name)
    rec=read(BEST/'resultado.json')
    assert sha(BEST/'modelo.pt')==rec['model_sha256']
    tr,X,names=feature_input('train',rec)
    va,V,vnames=feature_input('validation',rec)
    assert names==vnames==rec['features']
    y=np.load(DATA/'train_y.npy')
    v=np.load(DATA/'validation_y.npy')
    parent=OUT/'espacial_prior_cf_s0'
    parent_sha=sha(parent/'modelo.pt')
    ck=torch.load(parent/'modelo.pt',map_location='cpu',weights_only=False)
    head=torch.load(OUT/'cabeza_fases_cpu/cabeza.pt',map_location='cpu',weights_only=False)
    assert parent_sha==head['parent_sha256']
    assert head['features']==names==ck['features']
    grid=ck['grid']
    graph=dict(np.load(OUT/'datos_semi_markov/grafo_grid.npz'))
    distance=np.load(OUT/'datos_semi_markov/distancias_grid.npy')
    counts=np.load(OUT/'prior_transiciones/conteos_fold.npz')['counts']
    init=head['phase_initialization']
    model=PhaseDirectionalArrival(ck['stats'],grid,graph,distance,ck['power'],counts,init).cuda()
    shared=model.load_shared_state(ck['state'])
    model.wait_head.load_state_dict(head['head_state'])
    stats=ck['stats']
    power=ck['power']
    del counts,ck,head
    # Feature matrices and all CDF caches stay in RAM; no disk bank is generated.
    x=torch.tensor(X,device='cuda')
    vx=torch.tensor(V,device='cuda')
    del X,V
    target=torch.tensor(y,device='cuda')
    label=((target-60)/15).long()
    def ids(data,tag):
        xx=(data[tag+'_lon']+8.62)*111195*np.cos(np.deg2rad(41.16))
        yy=(data[tag+'_lat']-41.16)*111195
        a=np.floor((xx-grid['x0'])/200).astype(np.int64)
        b=np.floor((yy-grid['y0'])/200).astype(np.int64)
        assert np.all((a>=0)&(a<grid['nx'])&(b>=0)&(b<grid['ny']))
        return torch.tensor(a*grid['ny']+b,device='cuda')
    origin,destination=ids(tr,'o'),ids(tr,'d')
    vo,vd=ids(va,'o'),ids(va,'d')
    folds=torch.tensor(((tr['timestamp']//(7*86400))%5).astype(np.int64),device='cuda')
    eligible=torch.tensor(np.flatnonzero((tr['timestamp']+y)<va['timestamp'].min()),device='cuda')
    original=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False)
    teacher=make_duration_model(original,rec['config']).cuda().eval()
    teacher.load_state_dict(original['state'])
    del original
    @torch.no_grad()
    def reference_cache(features,tag):
        result=torch.empty((len(features),116),device='cuda')
        torch.set_float32_matmul_precision('highest')
        for first in range(0,len(features),2048):
            block=features[first:first+2048]
            n=len(block)
            if n<2048:block=torch.cat([block,block[-1:].expand(2048-n,-1)])
            result[first:first+n]=teacher(block)[0].exp().mean(1)[:n]
            if first%131072==0:print(json.dumps(dict(stage='reference_in_RAM',split=tag,completed=first,total=len(features))),flush=True)
        assert torch.isfinite(result).all()
        torch.set_float32_matmul_precision('high')
        return result
    reference_train=reference_cache(x,'train')
    reference_val=reference_cache(vx,'Validation')
    del teacher
    torch.cuda.empty_cache()
    old=np.load(BEST/'validacion.npz')
    reference_pred=model.median(reference_val).cpu().numpy()
    assert np.array_equal(old['row'],va['row']) and np.array_equal(old['pred'],reference_pred)
    write(run/'verificacion_referencia.json',dict(passed=True,all127941_retained_predictions_exact=True,reference_sha256=rec['model_sha256'],no_distribution_bank_on_disk=True,shared_parent=parent_sha,shared_initialization=shared,Test_access=False))
    runs=np.load(OUT/'datos_semi_markov/runs_train.npz')
    cells=runs['cell'].astype(np.int32)
    duration=runs['duration_n'].astype(np.int64)
    offsets=runs['run_offsets']
    trip=runs['trip_train_index']
    valid=np.unpackbits(runs['outgoing_valid_bits'])[:len(cells)].astype(bool)
    rng=np.random.default_rng(2026)
    lags=torch.tensor(init['lags'],device='cuda')
    def local_loss():
        ri=rng.integers(0,len(cells),64)
        ti=np.searchsorted(offsets,ri,side='right')-1
        ii=torch.tensor(trip[ti].astype(np.int64),device='cuda')
        cur=torch.tensor(cells[ri].astype(np.int64),device='cuda')
        previous=torch.tensor(cells[np.maximum(ri-1,0)].astype(np.int64),device='cuda')
        nxt=torch.tensor(cells[np.minimum(ri+1,len(cells)-1)].astype(np.int64),device='cuda')
        terminal=torch.tensor(ri==offsets[ti+1]-1,device='cuda')
        known=torch.tensor((ri>offsets[ti])&(cells[np.maximum(ri-1,0)]<model.N)&valid[np.maximum(ri-1,0)],device='cuda')
        heading=torch.where(known,heading_ids(cur//grid['ny']-previous//grid['ny'],cur%grid['ny']-previous%grid['ny']),4)
        safe=torch.where(cur<model.N,cur,origin[ii])
        q,oo,dd,weak,e,we=model.encode_od(x[ii],origin[ii],destination[ii])
        p,w,h=model.phase_parameters(q,x[ii,:4],safe,dd,heading,folds[ii])
        e,we=model.end_parameters(e,we,heading)
        dx=nxt//grid['ny']-cur//grid['ny']
        dy=nxt%grid['ny']-cur%grid['ny']
        code=(dx+3)*7+dy+3
        action=(code-(code>24).long()).clamp(0,47)
        good=torch.tensor(valid[ri],device='cuda')&(~terminal)&(cur<model.N)&(nxt<model.N)&model.allowed[safe,action]
        dur=torch.tensor(duration[ri],device='cuda')
        exponent=dur.clamp(1,119)[:,None]-lags[None]-1
        lp=w.clamp_min(1e-30).log()+h.log()+exponent.clamp_min(0)*torch.log1p(-h)
        residence=-torch.logsumexp(lp.masked_fill(exponent<0,-torch.inf),-1)
        route=-p.gather(1,action[:,None])[:,0].clamp_min(1e-9).log()
        move=((route+residence)*good).sum()/good.sum().clamp_min(1)
        terminal_mask=terminal&(cur<model.N)
        end_time=-we.gather(1,dur.clamp(0,119)[:,None])[:,0].clamp_min(1e-9).log()
        end_time=(end_time*terminal_mask).sum()/terminal_mask.sum().clamp_min(1)
        at_dest=(cur==dd)&(cur<model.N)
        ep=e.clamp(1e-6,1-1e-6)
        end_event=-(terminal.float()*ep.log()+(~terminal).float()*torch.log1p(-ep))
        end_event=(end_event*at_dest).sum()/at_dest.sum().clamp_min(1)
        return move+end_time+end_event
    proxy=np.sort(np.random.default_rng(1729).choice(len(v),4096,replace=False))
    np.save(run/'selection_indices.npy',proxy)
    write(run/'baseline_selection.json',dict(n=len(proxy),retained=metric(v[proxy],reference_pred[proxy]),full_retained=metric(v,reference_pred)))
    @torch.no_grad()
    def predict(indices=None):
        model.eval()
        torch.set_float32_matmul_precision('highest')
        use=np.arange(len(v)) if indices is None else np.asarray(indices)
        outputs=[[],[],[]]
        for first in range(0,len(use),32):
            ix=torch.tensor(use[first:first+32],device='cuda')
            n=len(ix)
            if n<32:ix=torch.cat([ix,ix[-1:].expand(32-n)])
            p,mass,ref,path=model.posterior(vx[ix],vo[ix],vd[ix],reference_probability=reference_val[ix])
            graph_pred=torch.where(mass>1e-6,model.median(path),model.median(ref))
            for out,value in zip(outputs,[model.median(p),graph_pred,mass]):out.append(value[:n].cpu().numpy())
            if indices is None and first%4096==0:print(json.dumps(dict(stage='full_validation_prediction',completed_rows=first,total_rows=len(use))),flush=True)
        torch.set_float32_matmul_precision('high')
        return tuple(np.concatenate(parts) for parts in outputs)
    # Verify actual data gradients before allocating a long training budget.
    model.train()
    trial=local_loss()
    assert torch.isfinite(trial)
    trial.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    write(run/'verificacion_inicial.json',dict(passed=True,finite_local_gradients=True,reference_reproduced_exactly=True,Test_access=False))
    for p in model.od_head.parameters():p.requires_grad_(False)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001,weight_decay=.0003)
    schedule=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=3,min_lr=1e-5)
    history=[]
    best=float('inf')
    best_step=0
    bad=0
    def save(step,score):
        torch.save(dict(state=model.state_dict(),stats=stats,power=power,grid=grid,features=names,config=cfg,phase_initialization=init,parent_sha256=parent_sha,reference_sha256=rec['model_sha256']),run/'modelo.pt')
        write(run/'mejor.json',dict(step=step,selection_validation=score,seconds=time.time()-start))
    resume_file=run/'ultima_reanudacion.pt'
    first_step=0
    previous_seconds=0.
    finished=False
    if args.resume:
        saved=torch.load(resume_file,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        schedule.load_state_dict(saved['scheduler'])
        first_step=saved['step'];best=saved['best'];best_step=saved['best_step'];bad=saved['bad'];history=saved['history'];previous_seconds=saved['elapsed'];finished=saved['finished']
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        rng.bit_generator.state=saved['numpy_generator']
        del saved
    def save_resume(step,finished=False):
        temporary=run/'reanudacion.tmp'
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=schedule.state_dict(),step=step,best=best,best_step=best_step,bad=bad,history=history,elapsed=previous_seconds+time.time()-start,finished=finished,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_generator=rng.bit_generator.state),temporary)
        os.replace(temporary,resume_file)
    last_step=first_step
    training_steps=[] if finished else range(first_step+1,args.steps+1)
    for step in training_steps:
        model.train()
        opt.zero_grad(set_to_none=True)
        total=0.
        for micro in range(2):
            ii=eligible[torch.randint(len(eligible),(8,),device='cuda')]
            p,mass,ref,path=model.posterior(x[ii],origin[ii],destination[ii],excluded_fold=folds[ii],reference_probability=reference_train[ii])
            truth=(model.duration_grid[None]>=target[ii,None]).float()
            mixed=-p.gather(1,label[ii,None]).clamp_min(1e-9).log()[:,0]+5*(p.cumsum(-1)-truth).square().mean(-1)
            structural=-path.gather(1,label[ii,None]).clamp_min(1e-12).log()[:,0]+5*(path.cumsum(-1)-truth).square().mean(-1)
            supported=mass>1e-6
            value=mixed.mean()+.5*(structural*supported).sum()/supported.sum().clamp_min(1)
            assert torch.isfinite(value)
            (value/2).backward()
            total+=float(value.detach())/2
        auxiliary=local_loss()
        assert torch.isfinite(auxiliary)
        (.1*auxiliary).backward()
        if step==first_step+1:assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        opt.step()
        if step%100==0:print(json.dumps(dict(stage='train_marginal_total',step=step,total_queries=step*16,loss=total,auxiliary=float(auxiliary.detach()),seconds=time.time()-start)),flush=True)
        if step==1 or step%500==0:
            pp,gg,mm=predict(proxy)
            score=metric(v[proxy],pp)
            schedule.step(score['mae_s'])
            r=dict(step=step,train_total_queries=step*16,selection_validation=score,selection_graph=metric(v[proxy],gg),mean_mass=float(mm.mean()),loss=total,seconds=time.time()-start)
            history.append(r)
            write(run/'historia.json',history)
            print(json.dumps(r),flush=True)
            if score['mae_s']<best-.025:
                best=score['mae_s'];bad=0;best_step=step;save(step,score)
            else:bad+=1
        last_step=step
        if step==1 or step%500==0:save_resume(step)
        if (step>=3000 and bad>=6) or previous_seconds+time.time()-start>5400:break
    save_resume(last_step,finished=True)
    ck=torch.load(run/'modelo.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(ck['state'])
    pred,graph_pred,mass=predict()
    np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=v,pred=pred,reference=reference_pred,graph_with_reference_fallback=graph_pred,absorption_mass=mass)
    result=dict(id=args.id,config=cfg,validation=metric(v,pred),graph_validation=metric(v,graph_pred),reference_validation=metric(v,reference_pred),best_step=best_step,steps_run=last_step,total_queries_seen=last_step*16,local_training_states_seen=last_step*64,model_sha256=sha(run/'modelo.pt'),code_sha256=sha(__file__),model_code_sha256=sha(OUT/'modelo_dir_fases.py'),kernel_code_sha256=sha(OUT/'nucleo_fases.py'),reference_sha256=rec['model_sha256'],parent_sha256=parent_sha,head_initialization_sha256=sha(OUT/'cabeza_fases_cpu/cabeza.pt'),prior_counts_sha256=sha(OUT/'prior_transiciones/conteos_fold.npz'),paired_vs_retained=paired(v,pred,reference_pred,va['timestamp']),actual_data_files_accessed=sorted(access),seconds=previous_seconds+time.time()-start,Test_access=False,new_primary_best=metric(v,pred)['mae_s']<metric(v,reference_pred)['mae_s'],physical_parameters=sum(p.numel() for p in model.parameters()),qualifies_for_preservation=metric(v,pred)['mae_s']<131.06 or metric(v,graph_pred)['mae_s']<131.06,goal_validation=metric(v,pred)['mae_s']<73.5)
    write(run/'resultado.json',result)
    print(json.dumps({k:result[k] for k in ['id','validation','graph_validation','reference_validation','best_step','steps_run','seconds']}),flush=True)

if __name__=='__main__':
    main()

