"""Protect every completed qualifying control checkpoint while training remains live."""
from base_replanteo import *
run=OUT/'control_historica_fresca_s0';history=read(run/'historia.json');candidates=[]
for item in history:
 if item['validation']['mae_s']>=131.06:continue
 path=(run/('epoca_'+str(item['epoch'])+'.pt')).resolve()
 assert path.parent==run.resolve() and path.is_file()
 before=path.stat();assert 0<before.st_size<64*1024**2
 digest=sha(path);after=path.stat();assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
 candidates.append(dict(epoch=item['epoch'],model_file=str(path),model_sha256=digest,validation=item['validation'],preserve=True))
dest=OUT/'candidatos_vivos_control_historica.json'
if dest.exists():
 previous=read(dest)
 for old in previous['candidates']:
  match=[c for c in candidates if c['epoch']==old['epoch']]
  assert len(match)==1 and match[0]['model_sha256']==old['model_sha256']
write(dest,dict(experiment=run.name,stage='Training active unless complete resultado.json exists; no certification claimed here.',
 candidates=candidates,latest_completed_epoch=history[-1]['epoch'],all_completed_metrics_predictions_retained=True,
 final_goal73_5_achieved=False,Test_access=False))
print(json.dumps(dict(qualifying_control_models_preserved=len(candidates),latest_completed_epoch=history[-1]['epoch'],best_observed_control_mae_s=min(q['validation']['mae_s'] for q in history))))
