"""Train a recurrent native15s generator, then evaluate unknown paths and ending times."""
from base_replanteo import *
from modelo_prefijo_historia import HistoricalPrefixArrival
from historia_eventos import TemporalEventHistory,event_groups
from datos_prefijo_nativo import NativeTrainSequences,query_seeds,prefix_loss
from modelo_mezcla_rutas import make_duration_model
from inferencia import feature_input
import torch,argparse,random,shutil

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--id',required=True)
    ap.add_argument('--epochs',type=int,default=6)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    access=deny_test()
    assert read(OUT/'verificacion_prefijo_historia.json')['passed']
    assert read(OUT/'verificacion_historia_eventos.json')['passed']
    pilot=read(OUT/'piloto_historia_condicional_s0/resultado.json')
    assert pilot['worth_GPU_test'],'Conditional pilot does not justify this GPU trial'
    assert not read(OUT/'piloto_entorno_compartido_s0/resultado.json')['worth_GPU_shared_effect_test']
    assert (OUT/'prefijo_nativo_s0/resultado.json').exists(),'Wait for complete native baseline before starting a new GPU job'
    torch.set_num_threads(4);torch.manual_seed(0);random.seed(0)
    torch.cuda.set_per_process_memory_fraction(.4,0)
    torch.set_float32_matmul_precision('high')
    run=OUT/args.id
    assert run.resolve().parent==OUT and (args.resume or not run.exists())
    run.mkdir(exist_ok=args.resume)
    (run/'fuentes').mkdir(exist_ok=args.resume)
    start=time.time()
    cfg=dict(id=args.id,seed=0,epochs=args.epochs,batch=128,lr=.0002,weight_decay=.0003,selection_n=512,selection_particles=128,full_particles=128,canonical_query_batch=128,reference_batch=2048,alphas=[0,.05,.2,.5,1],selection='Best alpha1 output with mass-dependent reference shrinkage on fixed512 Validation queries; alpha chosen only after checkpoint frozen',objective='Mean observed-sequence joint teacher-forced NLL scaled by constant120; actual prefixes used ONLY for Train supervision',query='216 OD/departure/static/history features, generated recurrent prefix and elapsed simulation time; NO true route, point count, elapsed trip observation or duration',generation='Fixed120 time steps, one interval15s; conditional non-stop particles with Rao-Blackwellized stopping mass at each step; cohort conditioning only after absorption',randomness='Stateless hash of exact raw OD/departure plus particle counter, step and stream; no row index and no input order RNG',prior='Route and temporal competing-risk histories exclude current Train weekfold at every level; Validation uses full Train only',event_head='New3-event residual head initialized from controlled CPU Train-only pilot; native stay/stop heads frozen and unused',inference_batch_scope='Canonical128 fixed before training; no cross-batch equality claim for stochastic paths. Validate query order and singleton at this canonical shape before promotion.',comparison_limit='Warm-start from completed native generator plus additional training; improvement alone is not an isolated causal ablation of the history prior',max_training_seconds=5400,minimum_epochs=3,patience=4,max_GPU_jobs=1,no_disk_distribution_or_hidden_bank=True,Test_access=False)
    if args.resume:
        cfg=read(run/'protocolo.json')
        assert cfg['epochs']==args.epochs and sha(__file__)==sha(run/'fuentes/entrenar_prefijo_historia.py')
    else:
        write(run/'protocolo.json',cfg)
        for name in ['modelo_prefijo_historia.py','historia_eventos.py','entrenar_prefijo_historia.py','modelo_prefijo_nativo.py','datos_prefijo_nativo.py','base_replanteo.py','modelo_dir_fases.py','nucleo_fases.py','modelo_dir_prior.py','modelo_direccional.py','modelo_espacial.py','nucleo_semi_markov.py','nucleo_direccional.py']:
            shutil.copy2(OUT/name,run/'fuentes'/name)
    rec=read(BEST/'resultado.json')
    assert sha(BEST/'modelo.pt')==rec['model_sha256']
    tr,X,names=feature_input('train',rec)
    va,V,vnames=feature_input('validation',rec)
    assert names==vnames==rec['features']
    y=np.load(DATA/'train_y.npy');v=np.load(DATA/'validation_y.npy')
    parent=OUT/'prefijo_nativo_s0'
    parent_sha=sha(parent/'modelo.pt')
    assert parent_sha==read(parent/'resultado.json')['model_sha256']
    ck=torch.load(parent/'modelo.pt',map_location='cpu',weights_only=False)
    grid=ck['grid'];stats=ck['stats'];power=ck['power'];init=ck['phase_initialization']
    graph=dict(np.load(OUT/'datos_semi_markov/grafo_grid.npz'))
    distance=np.load(OUT/'datos_semi_markov/distancias_grid.npy')
    counts=np.load(OUT/'prior_transiciones/conteos_fold.npz')['counts']
    event_meta=read(OUT/'prior_eventos/resultado.json')
    event_probability_sha=sha(OUT/'prior_eventos/probabilidad_train.npz')
    event_counts_sha=sha(OUT/'prior_eventos/conteos_fold.npz')
    assert event_probability_sha==event_meta['outputs']['probabilidad_train.npz']['sha256']
    assert event_counts_sha==event_meta['outputs']['conteos_fold.npz']['sha256']
    event_history=TemporalEventHistory(np.load(OUT/'prior_eventos/conteos_fold.npz')['counts'])
    full_event_probability=np.load(OUT/'prior_eventos/probabilidad_train.npz')['probability']
    model=HistoricalPrefixArrival(stats,grid,graph,distance,power,counts,init,full_event_probability).cuda()
    missing=model.load_native_initialization(ck['state'])
    head_path=OUT/'piloto_historia_condicional_s0/cabezas.pt'
    head_sha=sha(head_path)
    assert head_sha==pilot['head_sha256']
    head=torch.load(head_path,map_location='cpu',weights_only=False)
    assert head['parent_sha256']==parent_sha and head['worth_GPU_test']
    model.event_residual.load_state_dict(head['history_head'])
    del head
    del ck,counts,full_event_probability
    x=torch.tensor(X,device='cuda');vx=torch.tensor(V,device='cuda')
    del X,V
    def ids(data,tag):
        xx=(data[tag+'_lon']+8.62)*111195*np.cos(np.deg2rad(41.16))
        yy=(data[tag+'_lat']-41.16)*111195
        a=np.floor((xx-grid['x0'])/200).astype(np.int64)
        b=np.floor((yy-grid['y0'])/200).astype(np.int64)
        assert np.all((a>=0)&(a<grid['nx'])&(b>=0)&(b<grid['ny']))
        return a*grid['ny']+b
    origin,destination=ids(tr,'o'),ids(tr,'d')
    vo,vd=ids(va,'o'),ids(va,'d')
    origin_t=torch.tensor(origin,device='cuda');destination_t=torch.tensor(destination,device='cuda')
    vo_t=torch.tensor(vo,device='cuda');vd_t=torch.tensor(vd,device='cuda')
    folds=torch.tensor(((tr['timestamp']//(7*86400))%5).astype(np.int64),device='cuda')
    val_seeds=torch.tensor(query_seeds(va),device='cuda')
    val_timestamp=torch.tensor(va['timestamp'],device='cuda',dtype=torch.int64)
    sequences=NativeTrainSequences(OUT/'datos_semi_markov/runs_train.npz',graph['neighbors'],graph['allowed'],grid['ny'],model.N)
    eligible=np.flatnonzero((tr['timestamp'][sequences.trip]+y[sequences.trip])<va['timestamp'].min())
    assert len(eligible)==len(sequences.trip)==1023485
    old=np.load(BEST/'validacion.npz')
    assert np.array_equal(old['row'],va['row'])
    reference_pred=old['pred'].copy()
    original=torch.load(BEST/'modelo.pt',map_location='cpu',weights_only=False)
    teacher=make_duration_model(original,rec['config']).cuda().eval()
    teacher.load_state_dict(original['state'])
    reference=torch.empty((len(v),116),device='cuda')
    torch.set_float32_matmul_precision('highest')
    with torch.no_grad():
        for first in range(0,len(v),2048):
            block=vx[first:first+2048];n=len(block)
            if n<2048:block=torch.cat([block,block[-1:].expand(2048-n,-1)])
            reference[first:first+n]=teacher(block)[0].exp().mean(1)[:n]
    assert np.array_equal(model.median(reference).cpu().numpy(),reference_pred)
    del teacher,original
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    write(run/'verificacion_referencia.json',dict(passed=True,all127941_predictions_exact=True,reference_sha256=rec['model_sha256'],parent_sha256=parent_sha,new_parameters=missing,Test_access=False))
    proxy=np.sort(np.random.default_rng(1729).choice(len(v),512,replace=False))
    np.save(run/'selection_indices.npy',proxy)
    write(run/'baseline_selection.json',metric(v[proxy],reference_pred[proxy]))
    alphas=cfg['alphas']

    @torch.no_grad()
    def predict(indices,particles=128,stream=0,full=False):
        model.eval()
        torch.set_float32_matmul_precision('highest')
        outputs=np.empty((len(alphas),len(indices)),np.float32)
        masses=np.empty(len(indices),np.float32)
        for first in range(0,len(indices),128):
            ii=torch.tensor(indices[first:first+128],device='cuda');n=len(ii)
            if n<128:ii=torch.cat([ii,ii[-1:].expand(128-n)])
            raw,left=model.generated_arrival(vx[ii],vo_t[ii],vd_t[ii],val_seeds[ii],particles=particles,stream=stream,timestamp=val_timestamp[ii])
            mass=raw[:,4:120].sum(-1)
            assert torch.isfinite(raw).all() and (raw>=0).all() and ((raw.sum(-1)+left-1).abs()<2e-4).all()
            path=raw[:,4:120]/mass[:,None].clamp_min(1e-12)
            support=mass/(mass+.01)
            for j,alpha in enumerate(alphas):
                if alpha==0:
                    p=reference[ii]
                else:
                    a=alpha*support
                    p=(1-a[:,None])*reference[ii]+a[:,None]*path
                    p=p/p.sum(-1,keepdim=True)
                outputs[j,first:first+n]=model.median(p)[:n].cpu().numpy()
            masses[first:first+n]=mass[:n].cpu().numpy()
            if full and first%1024==0:print(json.dumps(dict(stage='full_native_generation',completed_rows=first,total_rows=len(indices),particles=particles)),flush=True)
        torch.set_float32_matmul_precision('high')
        return outputs,masses

    def batch_loss(sequence_indices):
        rows,nb=sequences.make(sequence_indices,origin,destination,y)
        ii=torch.tensor(rows,device='cuda')
        b={k:torch.tensor(value,device='cuda') for k,value in nb.items()}
        groups=event_groups(nb['cell'],nb['heading'],destination[rows,None],tr['timestamp'][rows,None],np.arange(120)[None,:],nb['age'])
        excluded=((tr['timestamp'][rows]//(7*86400))%5)[:,None]
        prior=event_history.lookup(groups,nb['age'],excluded)
        event_prior=torch.tensor(prior,device='cuda')
        motion,stop=model.teacher_probabilities(x[ii],origin_t[ii],destination_t[ii],b['cell'],b['heading'],b['age'],folds[ii],event_prior=event_prior)
        return prefix_loss(motion,stop,b)

    warmup,_=predict(proxy[:128])
    assert np.isfinite(warmup).all()
    assert np.array_equal(warmup[0],reference_pred[proxy[:128]])
    model.train()
    initial=batch_loss(eligible[:8])
    assert torch.isfinite(initial)
    initial.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    write(run/'verificacion_GPU_inicial.json',dict(passed=True,finite_actual_Train_gradients=True,reference_reproduced=True,canonical128_inference_warmup_passed=True,Test_access=False))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0002,weight_decay=.0003)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=2,min_lr=1e-5)
    rng=np.random.default_rng(2026)
    history=[];best=float('inf');best_epoch=0;bad=0;first_epoch=0;previous_seconds=0.
    steps=0
    resume_file=run/'ultima_reanudacion.pt'
    if args.resume:
        saved=torch.load(resume_file,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        first_epoch=saved['epoch'];best=saved['best'];best_epoch=saved['best_epoch'];bad=saved['bad'];history=saved['history'];steps=saved['steps'];previous_seconds=saved['elapsed']
        rng.bit_generator.state=saved['numpy_generator']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        del saved
    def save_resume(epoch):
        temp=run/'reanudacion.tmp'
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,best=best,best_epoch=best_epoch,bad=bad,history=history,steps=steps,elapsed=previous_seconds+time.time()-start,numpy_generator=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),temp)
        os.replace(temp,resume_file)
    last_epoch=first_epoch
    for epoch in range(first_epoch+1,args.epochs+1):
        order=rng.permutation(eligible)
        model.train()
        total=0.;seen=0
        for first in range(0,len(order),128):
            chosen=order[first:first+128]
            opt.zero_grad(set_to_none=True)
            loss=batch_loss(chosen)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            opt.step()
            total+=float(loss.detach())*len(chosen);seen+=len(chosen);steps+=1
            if steps%500==0:print(json.dumps(dict(stage='native_prefix_train',epoch=epoch,steps=steps,trips_in_epoch=seen,mean_NLL=total/seen,seconds=time.time()-start)),flush=True)
        pred,mass=predict(proxy)
        graph_metric=metric(v[proxy],pred[-1])
        selection=[metric(v[proxy],p) for p in pred]
        r=dict(epoch=epoch,steps=steps,mean_NLL=total/seen,selection_all_alphas=selection,selection_graph=graph_metric,mean_absorption_mass=float(mass.mean()),seconds=previous_seconds+time.time()-start)
        history.append(r);write(run/'historia.json',history);print(json.dumps(r),flush=True)
        scheduler.step(graph_metric['mae_s'])
        if graph_metric['mae_s']<best-.025:
            best=graph_metric['mae_s'];best_epoch=epoch;bad=0
            torch.save(dict(state=model.state_dict(),stats=stats,power=power,grid=grid,features=names,config=cfg,phase_initialization=init,parent_sha256=parent_sha,reference_sha256=rec['model_sha256'],event_probability_sha256=event_probability_sha,event_counts_sha256=event_counts_sha),run/'modelo.pt')
            write(run/'mejor.json',r)
        else:bad+=1
        save_resume(epoch);last_epoch=epoch
        if (epoch>=3 and bad>=4) or previous_seconds+time.time()-start>5400:break
    ck=torch.load(run/'modelo.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['state']);del ck
    del x,opt,scheduler,event_history
    for name in ['counts_fold','counts_base_fold','counts_global_fold']:setattr(model,name,torch.empty(0,device='cuda'))
    torch.cuda.empty_cache()
    selection,masses=predict(proxy)
    selected=int(np.argmin([metric(v[proxy],p)['mae_s'] for p in selection]))
    alt,_=predict(proxy,stream=1)
    stability=dict(particles=128,selection_n=len(proxy),selected_alpha=alphas[selected],stream0=metric(v[proxy],selection[selected]),stream1=metric(v[proxy],alt[selected]),mean_absolute_prediction_change_s=float(np.abs(selection[selected]-alt[selected]).mean()),selection_graph_stream0=metric(v[proxy],selection[-1]),selection_graph_stream1=metric(v[proxy],alt[-1]),interpretation='Monte Carlo sensitivity, not independent validation; seeds and128particles fixed before full evaluation. Any threshold candidate needs512particle robustness and raw/order reproduction.')
    write(run/'estabilidad_monte_carlo.json',stability)
    pred,mass=predict(np.arange(len(v)),full=True)
    assert np.array_equal(pred[0],reference_pred)
    all_metrics=[metric(v,p) for p in pred]
    np.savez_compressed(run/'validacion.npz',row=va['row'],timestamp=va['timestamp'],true=v,pred=pred,alphas=np.array(alphas),selected_alpha=alphas[selected],absorption_mass=mass)
    qualified=[dict(alpha=a,validation=m,robustness_512_and_reproduction_pending=True) for a,m in zip(alphas,all_metrics) if m['mae_s']<131.06]
    result=dict(id=args.id,config=cfg,validation=all_metrics[selected],graph_validation=all_metrics[-1],all_alpha_validation=all_metrics,selected_alpha=alphas[selected],reference_validation=all_metrics[0],best_epoch=best_epoch,epochs_run=last_epoch,steps=steps,model_sha256=sha(run/'modelo.pt'),code_sha256=sha(__file__),model_code_sha256=sha(OUT/'modelo_prefijo_historia.py'),data_code_sha256=sha(OUT/'datos_prefijo_nativo.py'),event_probability_sha256=event_probability_sha,event_counts_sha256=event_counts_sha,head_initialization_sha256=head_sha,history_code_sha256=sha(OUT/'historia_eventos.py'),parent_sha256=parent_sha,reference_sha256=rec['model_sha256'],paired_vs_retained=paired(v,pred[selected],reference_pred,va['timestamp']),qualified_candidates=qualified,new_primary_best_pending_reproduction=all_metrics[selected]['mae_s']<all_metrics[0]['mae_s'],goal_validation=all_metrics[selected]['mae_s']<73.5,actual_data_files_accessed=sorted(access),Test_access=False,seconds=previous_seconds+time.time()-start)
    write(run/'resultado.json',result)
    print(json.dumps({k:result[k] for k in ['id','validation','graph_validation','all_alpha_validation','selected_alpha','qualified_candidates','seconds']}),flush=True)

if __name__=='__main__':main()
