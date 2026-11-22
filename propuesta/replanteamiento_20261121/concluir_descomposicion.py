from base_replanteo import *
import shutil
access=deny_test();a=np.load(OUT/'descompuesto_aux1_s0/validacion.npz');b=np.load(OUT/'descompuesto_aux0_s0/validacion.npz');old=np.load(BEST/'validacion.npz');assert np.array_equal(a['row'],b['row']) and np.array_equal(a['row'],old['row']);assert np.array_equal(a['true'],b['true'])
report=dict(kind='Matched generative decomposition ablation; complete reused Validation',n=len(a['true']),supervised_proxy=metric(a['true'],a['pred']),same_architecture_without_proxy=metric(b['true'],b['pred']),retained=metric(old['true'],old['pred']),supervised_minus_control=paired(a['true'],a['pred'],b['pred'],a['timestamp']),control_minus_retained=paired(a['true'],b['pred'],old['pred'],a['timestamp']),scope='Single seed0, conditional-count auxiliary weight1 versus0, same cohorts/features/model family and scheduled maximum100 epochs with patience20',conclusion='No measured improvement from this low-displacement auxiliary target at weight1. Do not equate GPS low-displacement with annotated idling, and do not generalize to every pause model.',Test_access=False)
write(OUT/'comparacion_descomposicion.json',report)
run=OUT/'descompuesto_aux0_s0';r=read(run/'resultado.json');p=run/'modelo.pt'
if p.exists():
 assert run.resolve().parent==OUT and sha(p)==r['model_sha256'] and r['validation']['mae_s']>=131.06
 sources=run/'fuentes';sources.mkdir(exist_ok=True)
 for name in ['entrenar_descompuesto.py','modelo_descompuesto.py','base_replanteo.py']:
  assert name not in ('common.py',);dst=sources/name;assert not dst.exists() or sha(dst)==sha(OUT/name)
  if not dst.exists():shutil.copy2(OUT/name,dst)
 rec=dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p),predictions_sha256=sha(run/'validacion.npz'),mae_s=r['validation']['mae_s'],reason='Complete matched control does not improve131.06s; exact sources, protocol and predictions retained');p.unlink();write(run/'pesos_descartados.json',rec)
print(json.dumps({k:report[k] for k in ['supervised_proxy','same_architecture_without_proxy','supervised_minus_control','control_minus_retained']},default=lambda v:v.tolist()))

