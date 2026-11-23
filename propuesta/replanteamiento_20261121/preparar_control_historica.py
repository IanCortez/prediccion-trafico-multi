"""Prepare an unchanged-math control runner with bounded checkpoint retention."""
from base_replanteo import *
main=OUT/'historica_fresca_s0';reference=read(OUT/'nuevo_mejor_verificado.json');protocol=read(main/'protocolo.json')
assert reference['model_sha256']==protocol['reference_checkpoint_sha256']
write(main/'referencia_previa.json',reference)
source=main/'fuentes/entrenar_historica_fresca.py';code=source.read_text(encoding='utf-8-sig')
assert sha(source)==sha(OUT/'entrenar_historica_fresca.py')
code=code.replace("reference=read(OUT/'nuevo_mejor_verificado.json')","reference=read(OUT/'historica_fresca_s0/referencia_previa.json')")
code=code.replace("sources=['entrenar_historica_fresca.py'","sources=['entrenar_historica_fresca_control.py'")
needle="args=ap.parse_args()\naccess="
assert needle in code;code=code.replace(needle,"args=ap.parse_args()\nassert args.mode=='parent_control'\naccess=")
code=code.replace("all_pred={'reference':ref['pred']};history=[];saved=[];frontier=","all_pred={'reference':ref['pred']};history=[];saved=[];discarded=[];frontier=")
needle="m=metric(v,pred);save=m['mae_s']<frontier or m['mae_s']<131.06;path="
replacement="m=metric(v,pred);old_file=best_file;old_score=best;improves_run=m['mae_s']<best;save=improves_run or m['mae_s']<frontier or m['mae_s']<131.06;path="
assert needle in code;code=code.replace(needle,replacement)
needle="if m['mae_s']<best:best=m['mae_s'];best_epoch=epoch;best_file=str(path) if save else None\n return m,save"
replacement="""if improves_run:
  best=m['mae_s'];best_epoch=epoch;best_file=str(path)
  if old_file and old_file!=str(path) and old_score>=131.06 and old_score>=config['reference_mae_s']:
   previous=Path(old_file).resolve();assert previous.parent==run.resolve() and previous.stat().st_size<64*1024**2
   discarded.append(dict(file=previous.name,sha256=sha(previous),mae_s=old_score,reason='Superseded nonqualifying control checkpoint; every metric/prediction retained.'))
   previous.unlink();saved[:]=[entry for entry in saved if entry['file']!=previous.name]
 return m,save"""
assert needle in code;code=code.replace(needle,replacement)
code=code.replace("history=state['history'];saved=state['saved'];frontier=","history=state['history'];saved=state['saved'];discarded=state.get('discarded',[]);frontier=")
code=code.replace("history=history,saved=saved,frontier=frontier","history=history,saved=saved,discarded=discarded,frontier=frontier")
code=code.replace("model_file=best_file,model_sha256=sha(best_file) if best_file else None,saved_checkpoints=saved,","model_file=best_file,model_sha256=sha(best_file) if best_file else None,saved_checkpoints=saved,discarded_control_checkpoints=discarded,")
path=OUT/'entrenar_historica_fresca_control.py';assert not path.exists();path.write_text(code,encoding='utf-8')
import ast
ast.parse(code)
write(OUT/'control_historica_diferencias_IO.json',dict(control_source_sha256=sha(path),treatment_source_sha256=sha(source),
 math_optimizer_sampler_architecture_unchanged=True,
 differences=['Explicit frozen pretrial reference, so registry promotion cannot change comparison.','Keep current best control weights even if worse than reference for uncertainty/reproduction checks.','Delete only superseded nonqualifying control checkpoint files inside its own run, retaining all metrics/predictions and deletion hashes.'],
 all_full_MAE_below131_06_preserved=True,
 note='Separate new control runner; active treatment source and imports were not edited. No control launched while treatment GPU is live.'))
print('Prepared separate control runner; model math and training protocol unchanged.')
