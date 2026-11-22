from base_replanteo import *
import ast
source=OUT/'prefijo_nativo_s0/fuentes/entrenar_prefijo_nativo.py'
s=source.read_text(encoding='utf-8-sig')
def replace(a,b):
    global s
    assert a in s,a
    s=s.replace(a,b)
replace('from modelo_prefijo_nativo import NativePrefixArrival','from modelo_prefijo_historia import HistoricalPrefixArrival\nfrom historia_eventos import TemporalEventHistory,event_groups')
replace("default=10)","default=6)")
replace("assert read(OUT/'verificacion_prefijo_nativo.json')['passed']","assert read(OUT/'verificacion_prefijo_historia.json')['passed']\n    assert read(OUT/'verificacion_historia_eventos.json')['passed']")
replace("assert (OUT/'calibracion_unica_fases_s0/resultado.json').exists(),'Finish previous GPU job first'","assert (OUT/'prefijo_nativo_s0/resultado.json').exists(),'Wait for complete native baseline before starting a new GPU job'")
replace('entrenar_prefijo_nativo.py','entrenar_prefijo_historia.py')
replace("for name in ['entrenar_prefijo_historia.py',","for name in ['modelo_prefijo_historia.py','historia_eventos.py','entrenar_prefijo_historia.py',")
replace("parent=OUT/'espacial_fases_referencia_s0'","parent=OUT/'prefijo_nativo_s0'")
replace("model=NativePrefixArrival(stats,grid,graph,distance,power,counts,init).cuda()\n    missing=model.load_phase_initialization(ck['state'])\n    del ck,counts","event_meta=read(OUT/'prior_eventos/resultado.json')\n    event_probability_sha=sha(OUT/'prior_eventos/probabilidad_train.npz')\n    event_counts_sha=sha(OUT/'prior_eventos/conteos_fold.npz')\n    assert event_probability_sha==event_meta['outputs']['probabilidad_train.npz']['sha256']\n    assert event_counts_sha==event_meta['outputs']['conteos_fold.npz']['sha256']\n    event_history=TemporalEventHistory(np.load(OUT/'prior_eventos/conteos_fold.npz')['counts'])\n    full_event_probability=np.load(OUT/'prior_eventos/probabilidad_train.npz')['probability']\n    model=HistoricalPrefixArrival(stats,grid,graph,distance,power,counts,init,full_event_probability).cuda()\n    missing=model.load_native_initialization(ck['state'])\n    del ck,counts,full_event_probability")
replace("val_seeds=torch.tensor(query_seeds(va),device='cuda')","val_seeds=torch.tensor(query_seeds(va),device='cuda')\n    val_timestamp=torch.tensor(va['timestamp'],device='cuda',dtype=torch.int64)")
replace("particles=particles,stream=stream)","particles=particles,stream=stream,timestamp=val_timestamp[ii])")
replace("motion,stop=model.teacher_probabilities(x[ii],origin_t[ii],destination_t[ii],b['cell'],b['heading'],b['age'],folds[ii])","groups=event_groups(nb['cell'],nb['heading'],destination[rows,None],tr['timestamp'][rows,None],np.arange(120)[None,:],nb['age'])\n        excluded=((tr['timestamp'][rows]//(7*86400))%5)[:,None]\n        prior=event_history.lookup(groups,nb['age'],excluded)\n        event_prior=torch.tensor(prior,device='cuda')\n        motion,stop=model.teacher_probabilities(x[ii],origin_t[ii],destination_t[ii],b['cell'],b['heading'],b['age'],folds[ii],event_prior=event_prior)")
replace("del x,opt,scheduler","del x,opt,scheduler,event_history")
replace("data_code_sha256=sha(OUT/'datos_prefijo_nativo.py'),","data_code_sha256=sha(OUT/'datos_prefijo_nativo.py'),event_probability_sha256=event_probability_sha,event_counts_sha256=event_counts_sha,history_code_sha256=sha(OUT/'historia_eventos.py'),")
replace("model_code_sha256=sha(OUT/'modelo_prefijo_nativo.py')","model_code_sha256=sha(OUT/'modelo_prefijo_historia.py')")
replace("prior='Current Train weekfold counts excluded; Validation uses Train totals'","prior='Route and temporal competing-risk histories exclude current Train weekfold at every level; Validation uses full Train only',event_head='New3-event residual logits initialized0 around empirical prior; native stay/stop heads frozen and unused',comparison_limit='Warm-start from completed native generator plus additional training; improvement alone is not an isolated causal ablation of the history prior'")
replace("parent_sha256=parent_sha,reference_sha256=rec['model_sha256']),run/'modelo.pt'","parent_sha256=parent_sha,reference_sha256=rec['model_sha256'],event_probability_sha256=event_probability_sha,event_counts_sha256=event_counts_sha),run/'modelo.pt'")
ast.parse(s)
target=OUT/'entrenar_prefijo_historia.py'
assert not target.exists()
target.write_text(s,encoding='utf-8')
write(OUT/'preparacion_entrenador_historia.json',dict(source=str(source),source_sha256=sha(source),target_sha256=sha(target),not_started=True,guards='Complete native result and CPU verification required; only one GPU job',Test='sealed'))
print('Entrenador con historia temporal preparado; todavía no iniciado.')
