"""On-policy full-trip conditional CRPS, with teacher regularization and unchanged OD cohort."""
from base_replanteo import *
from modelo_prefijo_nativo import NativePrefixArrival
from modelo_prefijo_historia import HistoricalPrefixArrival
from datos_prefijo_nativo import NativeTrainSequences,query_seeds,prefix_loss
from historia_eventos import TemporalEventHistory,event_groups
from muestreador_rechazo_nativo import sample_cohort_trajectories
from gradiente_crps_condicional import conditional_crps_surrogate
from modelo_mezcla_rutas import make_duration_model
from inferencia import feature_input
import torch,argparse,random,shutil

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--id',required=True);ap.add_argument('--steps',type=int,default=4000);ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    access=deny_test()
    assert read(OUT/'verificacion_muestreador_rechazo.json')['passed']
    assert read(OUT/'sondeo_riesgo_total_cpu.json')['passed']
    assert (OUT/'prefijo_nativo_s0/resultado.json').exists()
    hp=read(OUT/'piloto_historia_condicional_s0/resultado.json')
    if hp['worth_GPU_test']:assert (OUT/'prefijo_historia_s0/resultado.json').exists(),'Complete preceding history trial first'
    parents=[OUT/'prefijo_nativo_s0']
    if (OUT/'prefijo_historia_s0/resultado.json').exists():parents.append(OUT/'prefijo_historia_s0')
    parent=min(parents,key=lambda p:read(p/'resultado.json')['graph_validation']['mae_s'])
    for p in parents:
        assert min(m['mae_s'] for m in read(p/'resultado.json')['all_alpha_validation'])>=73.5,'Freeze eligible candidate instead of more research'
    torch.set_num_threads(4);torch.manual_seed(0);random.seed(0)
    torch.cuda.set_per_process_memory_fraction(.4,0)
    torch.set_float32_matmul_precision('high')
    run=OUT/args.id
    assert run.resolve().parent==OUT and (args.resume or not run.exists())
    run.mkdir(exist_ok=args.resume);(run/'fuentes').mkdir(exist_ok=args.resume)
    start=time.time()
    cfg=dict(id=args.id,seed=0,steps=args.steps,microbatch=4,accumulation=4,effective_queries=16,accepted_samples=8,candidates_per_round=16,max_rejection_rounds=8,teacher_batch=32,teacher_weight=.1,lr=.00001,weight_decay=.0003,clip_norm=1,score_scale_s=600,selection_n=512,selection_interval=500,patience=5,minimum_steps=1500,max_training_seconds=7200,canonical_query_batch=128,particles=128,reference_batch=2048,alphas=[0,.05,.2,.5,1],parent=str(parent),parent_rule='Lowest completed structural alpha1 MAE, with mass-dependent reference shrinkage; adaptive Validation selection, not Test',objective='Score-function covariance U-statistic for conditional full-trip CRPS +.1teacher-forced jointNLL/120',sampler='IID complete trajectories; first8 accepted only under fixed60..1785 event; full unconditional path log probabilities. Query label enters risk only after sampling.',support_failure='Teacher loss on failed query replaces its missing policy contribution; no query removed from primary evaluation. Report failure rate; do not claim exact all-query risk gradient under finite-budget fallback.',encoder_dropout='Disabled for policy sampling; GRU training mode with dropout0 enables CUDA backward. Teacher regularization uses ordinary training dropout.',history='Every route and optional temporal count excludes query Train weekfold; all eligible Train for Validation',selection='Best alpha1 output on fixed512; alpha selected after checkpoint frozen; alpha0 is the old reference',numerics='TrainingFP32high; inferenceFP32highest, canonical128 fixed before trial. Not cross-batch-equivalent to native16.',max_GPU_jobs=1,no_disk_distribution_or_hidden_bank=True,Test_access=False)
    if args.resume:
        cfg=read(run/'protocolo.json')
        assert cfg['steps']==args.steps and sha(__file__)==sha(run/'fuentes/entrenar_riesgo_total.py')
        parent=Path(cfg['parent'])
    else:
        write(run/'protocolo.json',cfg)
        for name in ['entrenar_riesgo_total.py','muestreador_rechazo_nativo.py','gradiente_crps_condicional.py','modelo_prefijo_nativo.py','modelo_prefijo_historia.py','datos_prefijo_nativo.py','historia_eventos.py','base_replanteo.py','modelo_dir_fases.py','nucleo_fases.py','modelo_dir_prior.py','modelo_direccional.py','modelo_espacial.py','nucleo_direccional.py','nucleo_semi_markov.py']:shutil.copy2(OUT/name,run/'fuentes'/name)
    parent_record=read(parent/'resultado.json')
    parent_sha=sha(parent/'modelo.pt')
    assert parent_sha==parent_record['model_sha256']
    ck=torch.load(parent/'modelo.pt',map_location='cpu',weights_only=False)
    grid=ck['grid'];stats=ck['stats'];power=ck['power'];init=ck['phase_initialization']
    graph=dict(np.load(OUT/'datos_semi_markov/grafo_grid.npz'))
    distance=np.load(OUT/'datos_semi_markov/distancias_grid.npy')
    counts=np.load(OUT/'prior_transiciones/conteos_fold.npz')['counts']
    historical='event_residual.weight' in ck['state']
    event_history=None;event_probability_sha=None;event_counts_sha=None
    if historical:
        event_probability_sha=sha(OUT/'prior_eventos/probabilidad_train.npz')
        event_counts_sha=sha(OUT/'prior_eventos/conteos_fold.npz')
        assert event_probability_sha==parent_record['event_probability_sha256']
        assert event_counts_sha==parent_record['event_counts_sha256']
        event_history=TemporalEventHistory(np.load(OUT/'prior_eventos/conteos_fold.npz')['counts'])
        full=np.load(OUT/'prior_eventos/probabilidad_train.npz')['probability']
        model=HistoricalPrefixArrival(stats,grid,graph,distance,power,counts,init,full).cuda()
        del full
    else:model=NativePrefixArrival(stats,grid,graph,distance,power,counts,init).cuda()
    model.load_state_dict(ck['state']);del ck,counts
    rec=read(BEST/'resultado.json')
    assert sha(BEST/'modelo.pt')==rec['model_sha256']
    tr,X,names=feature_input('train',rec);va,V,vnames=feature_input('validation',rec)
    assert names==vnames==rec['features']
    y=np.load(DATA/'train_y.npy');v=np.load(DATA/'validation_y.npy')
    x=torch.tensor(X,device='cuda');vx=torch.tensor(V,device='cuda');del X,V
    target=torch.tensor(y,device='cuda')
    def ids(d,tag):
        xx=(d[tag+'_lon']+8.62)*111195*np.cos(np.deg2rad(41.16))
        yy=(d[tag+'_lat']-41.16)*111195
        return np.floor((xx-grid['x0'])/200).astype(np.int64)*grid['ny']+np.floor((yy-grid['y0'])/200).astype(np.int64)
    origin,destination=ids(tr,'o'),ids(tr,'d')
    o=torch.tensor(origin,device='cuda');d=torch.tensor(destination,device='cuda')
    vo=torch.tensor(ids(va,'o'),device='cuda');vd=torch.tensor(ids(va,'d'),device='cuda')
    timestamp=torch.tensor(tr['timestamp'],device='cuda',dtype=torch.int64)
    vtimestamp=torch.tensor(va['timestamp'],device='cuda',dtype=torch.int64)
    folds=(timestamp//(7*86400))%5
    val_seeds=torch.tensor(query_seeds(va),device='cuda')
    sequences=NativeTrainSequences(OUT/'datos_semi_markov/runs_train.npz',graph['neighbors'],graph['allowed'],grid['ny'],model.N)
    eligible=np.flatnonzero(tr['timestamp'][sequences.trip]+y[sequences.trip]<va['timestamp'].min())
    assert len(eligible)==1023485
    old=np.load(BEST/'validacion.npz')
    assert np.array_equal(old['row'],va['row'])
    reference_pred=old['pred']
    original=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False)
    teacher=make_duration_model(original,rec['config']).cuda().eval();teacher.load_state_dict(original['state'])
    reference=torch.empty((len(v),116),device='cuda')
    torch.set_float32_matmul_precision('highest')
    with torch.no_grad():
        for first in range(0,len(v),2048):
            block=vx[first:first+2048];n=len(block)
            if n<2048:block=torch.cat([block,block[-1:].expand(2048-n,-1)])
            reference[first:first+n]=teacher(block)[0].exp().mean(1)[:n]
    assert np.array_equal(model.median(reference).cpu().numpy(),reference_pred)
    del teacher,original;torch.cuda.empty_cache();torch.set_float32_matmul_precision('high')
    write(run/'verificacion_referencia.json',dict(passed=True,all127941_exact=True,reference_sha256=rec['model_sha256'],parent_sha256=parent_sha,historical_parent=historical,Test_access=False))
    rng=np.random.default_rng(2026)
    proxy=np.sort(np.random.default_rng(1729).choice(len(v),512,replace=False));np.save(run/'selection_indices.npy',proxy)
    alphas=cfg['alphas']
    def teacher_loss(sequence_indices):
        rows,b=sequences.make(sequence_indices,origin,destination,y)
        ii=torch.tensor(rows,device='cuda')
        tb={k:torch.tensor(a,device='cuda') for k,a in b.items()}
        kwargs={}
        if historical:
            g=event_groups(b['cell'],b['heading'],destination[rows,None],tr['timestamp'][rows,None],np.arange(120)[None],b['age'])
            f=((tr['timestamp'][rows]//(7*86400))%5)[:,None]
            kwargs['event_prior']=torch.tensor(event_history.lookup(g,b['age'],f),device='cuda')
        motion,stop=model.teacher_probabilities(x[ii],o[ii],d[ii],tb['cell'],tb['heading'],tb['age'],folds[ii],**kwargs)
        return prefix_loss(motion,stop,tb)
    def draw(sequence_indices,particles=8,candidates=16):
        rows=sequences.trip[sequence_indices]
        ii=torch.tensor(rows,device='cuda')
        model.eval();model.prefix_rnn.train()
        sample=sample_cohort_trajectories(model,x[ii],o[ii],d[ii],particles=particles,candidates=candidates,max_rounds=8,fold=folds[ii],timestamp=timestamp[ii],event_history=event_history)
        return ii,sample
    @torch.no_grad()
    def predict(indices,full=False):
        model.eval();torch.set_float32_matmul_precision('highest')
        pred=np.empty((len(alphas),len(indices)),np.float32);mass=np.empty(len(indices),np.float32)
        for first in range(0,len(indices),128):
            ix=torch.tensor(indices[first:first+128],device='cuda');n=len(ix)
            if n<128:ix=torch.cat([ix,ix[-1:].expand(128-n)])
            kwargs={'timestamp':vtimestamp[ix]} if historical else {}
            raw,left=model.generated_arrival(vx[ix],vo[ix],vd[ix],val_seeds[ix],particles=128,**kwargs)
            assert torch.isfinite(raw).all() and ((raw.sum(-1)+left-1).abs()<2e-4).all()
            mm=raw[:,4:120].sum(-1);path=raw[:,4:120]/mm[:,None].clamp_min(1e-12)
            for j,alpha in enumerate(alphas):
                if alpha==0:p=reference[ix]
                else:
                    aa=alpha*mm/(mm+.01)
                    p=(1-aa[:,None])*reference[ix]+aa[:,None]*path
                    p=p/p.sum(-1,keepdim=True)
                pred[j,first:first+n]=model.median(p)[:n].cpu().numpy()
            mass[first:first+n]=mm[:n].cpu().numpy()
            if full and first%4096==0:print(json.dumps(dict(stage='full_risk_validation',completed_rows=first,total_rows=len(indices))),flush=True)
        torch.set_float32_matmul_precision('high')
        return pred,mass
    # GPU differentiation through complete sampled paths is a mandatory gate.
    smoke_sequences=np.random.default_rng(2026).choice(eligible,4,replace=False)
    ii,smoke=draw(smoke_sequences,particles=4,candidates=8)
    assert smoke['complete'].any()
    ok=smoke['complete']
    value,_=conditional_crps_surrogate(smoke['times'][ok].double(),target[ii][ok].double(),smoke['log_probability'][ok])
    value.backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.norm()) for g in gradients)>0
    model.zero_grad(set_to_none=True)
    del smoke,value,gradients
    initial,imass=predict(proxy)
    monitor_sequences=np.sort(np.random.default_rng(314).choice(eligible,256,replace=False))
    model.eval()
    with torch.no_grad():initial_teacher=float(teacher_loss(monitor_sequences))
    write(run/'verificacion_GPU_inicial.json',dict(passed=True,finite_nonzero_complete_path_gradients=True,smoke_Train_rows=sequences.trip[smoke_sequences].tolist(),encoder_dropout_disabled=True,GRU_train_mode_dropout0=True,canonical128_inference_passed=True,initial_teacher_NLL=initial_teacher,Test_access=False))
    write(run/'baseline_selection.json',dict(reference=metric(v[proxy],initial[0]),initial_parent_same_canonical_graph=metric(v[proxy],initial[-1]),initial_mass=float(imass.mean())))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.00001,weight_decay=.0003)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=2,min_lr=1e-6)
    history=[];best=metric(v[proxy],initial[-1])['mae_s'];best_step=0;bad=0;first_step=0;previous_seconds=0.;failed_total=0;query_total=0;finished=False
    def save(step):
        torch.save(dict(state=model.state_dict(),stats=stats,power=power,grid=grid,features=names,phase_initialization=init,parent_sha256=parent_sha,reference_sha256=rec['model_sha256'],config=cfg,event_probability_sha256=event_probability_sha,event_counts_sha256=event_counts_sha),run/'modelo.pt')
        write(run/'mejor.json',dict(step=step,selection_graph_MAE_s=best,historical_parent=historical))
    if not args.resume:save(0)
    resume_file=run/'ultima_reanudacion.pt'
    if args.resume:
        saved=torch.load(resume_file,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        first_step=saved['step'];best=saved['best'];best_step=saved['best_step'];bad=saved['bad'];history=saved['history'];previous_seconds=saved['elapsed']
        rng.bit_generator.state=saved['numpy_generator'];torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        failed_total=saved['failed_total'];query_total=saved['query_total'];finished=saved['finished']
        del saved
    def save_resume(step,finished=False):
        temp=run/'reanudacion.tmp'
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),step=step,best=best,best_step=best_step,bad=bad,history=history,elapsed=previous_seconds+time.time()-start,failed_total=failed_total,query_total=query_total,finished=finished,numpy_generator=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),temp)
        os.replace(temp,resume_file)
    last_step=first_step
    training_steps=[] if finished else range(first_step+1,args.steps+1)
    for step in training_steps:
        opt.zero_grad(set_to_none=True);risk=0.;surrogate=0.;failed_step=0;attempts=0
        for micro in range(4):
            si=rng.choice(eligible,4,replace=False)
            ii,sample=draw(si)
            ok=sample['complete'];n=int(ok.sum())
            if n:
                loss,score=conditional_crps_surrogate(sample['times'][ok].double(),target[ii][ok].double(),sample['log_probability'][ok])
                (loss*(n/4)/4).backward()
                surrogate+=float(loss.detach())/4;risk+=float(score)*600*n
            fail=(~ok).cpu().numpy()
            failed_step+=int(fail.sum());attempts+=sample['candidate_attempts_per_query']
            if fail.any():
                model.train()
                fallback=teacher_loss(si[fail])
                (fallback*(fail.sum()/4)/4).backward()
            del sample
        model.train()
        aux=teacher_loss(rng.choice(eligible,32,replace=False))
        (.1*aux).backward()
        assert torch.isfinite(aux)
        if step==first_step+1:assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
        opt.step()
        failed_total+=failed_step;query_total+=16
        last_step=step
        if step%50==0:print(json.dumps(dict(stage='on_policy_full_trip_CRPS',step=step,queries=step*16,sampled_CRPS_mean_over_complete_queries_s=(risk/(16-failed_step) if failed_step<16 else None),complete_policy_queries=16-failed_step,score_surrogate=surrogate,teacher_NLL=float(aux.detach()),support_failures_step=failed_step,support_failure_rate=failed_total/query_total,attempted_candidates_per_micro_mean=attempts/4,seconds=time.time()-start)),flush=True)
        stop_drift=False
        if step%500==0:
            pp,mm=predict(proxy)
            model.eval()
            with torch.no_grad():local=float(teacher_loss(monitor_sequences))
            score=metric(v[proxy],pp[-1])
            scheduler.step(score['mae_s'])
            r=dict(step=step,queries=step*16,selection_all_alphas=[metric(v[proxy],p) for p in pp],selection_graph=score,teacher_monitor=local,initial_teacher_monitor=initial_teacher,support_failure_rate=failed_total/query_total,mean_mass=float(mm.mean()),seconds=previous_seconds+time.time()-start)
            history.append(r);write(run/'historia.json',history);print(json.dumps(r),flush=True)
            if score['mae_s']<best-.025:best=score['mae_s'];best_step=step;bad=0;save(step)
            else:bad+=1
            stop_drift=local>1.25*initial_teacher
        if step%250==0:save_resume(step)
        if stop_drift or (step>=1500 and bad>=5) or previous_seconds+time.time()-start>7200:break
    save_resume(last_step,finished=True)
    ck=torch.load(run/'modelo.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['state']);del ck
    del x,opt,scheduler,event_history
    for name in ['counts_fold','counts_base_fold','counts_global_fold']:setattr(model,name,torch.empty(0,device='cuda'))
    torch.cuda.empty_cache()
    pp,mm=predict(proxy)
    selected=int(np.argmin([metric(v[proxy],p)['mae_s'] for p in pp]))
    pred,mass=predict(np.arange(len(v)),full=True)
    assert np.array_equal(pred[0],reference_pred)
    metrics=[metric(v,p) for p in pred]
    np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=v,pred=pred,alphas=np.array(alphas),selected_alpha=alphas[selected],absorption_mass=mass)
    qualified=[dict(alpha=a,validation=m,reproduction_and_512particle_robustness_pending=True) for a,m in zip(alphas,metrics) if a>0 and m['mae_s']<131.06]
    result=dict(id=args.id,config=cfg,validation=metrics[selected],graph_validation=metrics[-1],all_alpha_validation=metrics,selected_alpha=alphas[selected],reference_validation=metrics[0],best_step=best_step,steps_run=last_step,learned_update_selected=best_step>0,model_sha256=sha(run/'modelo.pt'),code_sha256=sha(__file__),sampler_sha256=sha(OUT/'muestreador_rechazo_nativo.py'),estimator_sha256=sha(OUT/'gradiente_crps_condicional.py'),parent_sha256=parent_sha,parent_kind='history' if historical else 'native',reference_sha256=rec['model_sha256'],event_probability_sha256=event_probability_sha,event_counts_sha256=event_counts_sha,paired_vs_retained=paired(v,pred[selected],reference_pred,va['timestamp']),qualified_candidates=qualified,goal_validation=metrics[selected]['mae_s']<73.5,new_primary_best_pending_reproduction=metrics[selected]['mae_s']<metrics[0]['mae_s'],support_failure_rate=failed_total/max(query_total,1),actual_data_files_accessed=sorted(access),Test_access=False,seconds=previous_seconds+time.time()-start)
    write(run/'resultado.json',result)
    print(json.dumps({k:result[k] for k in ['id','validation','graph_validation','all_alpha_validation','best_step','qualified_candidates','seconds']}),flush=True)

if __name__=='__main__':main()
