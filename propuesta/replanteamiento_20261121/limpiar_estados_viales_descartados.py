"""Remove exact regenerable targets of the failed value trial, retaining every prediction and source."""
from base_replanteo import *
run=OUT/'valor_vial_parcial_s0';r=read(run/'resultado.json')
assert r['validation']['mae_s']>131.06 and not r['saved_checkpoints'] and not r['control_trigger_met']
p=OUT/'estados_valor_vial_s0/estados_Train.npz';meta=read(p.parent/'resultado.json');expected=meta['targets_sha256']
assert p.resolve().is_relative_to(OUT.resolve()) and p.is_file() and sha(p)==expected
size=p.stat().st_size
assert size==round(meta['artifact_MiB']*1024**2)
entry=dict(file=str(p.relative_to(ROOT)),bytes=size,sha256=expected,
 type='RegenerableTrain-onlyauxiliarytargets, not experimentpredictions or originaldata.',
 reason='Amortizedvaluefamilyfailed; noqueued/runningjobuses thisderivedsamplebank. Full232MiBTrainstreetcorpus and executedpreparer preserved.',
 reconstruction_source='estados_valor_vial_s0/fuente_ejecutada.py',full_corpus_manifest='datos_calles_parciales_Train_s0/resultado.json')
write(OUT/'limpieza_estados_viales_descartados.json',dict(removed=[entry],all_completed_predictions_metrics_preserved=True,originals_best_TransferTraj_unchanged=True))
p.unlink()
print(json.dumps(dict(removed_bytes=size,all_completed_predictions_preserved=True)),flush=True)
