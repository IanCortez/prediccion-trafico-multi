from base_replanteo import *
import torch
run=OUT/'prefijo_nativo_s0'
assert not (run/'resultado.json').exists()
saved=torch.load(run/'ultima_reanudacion.pt',map_location='cpu',weights_only=False)
assert saved['epoch']==10 and saved['steps']==79960
assert sha(OUT/'entrenar_prefijo_nativo.py')==sha(run/'fuentes/entrenar_prefijo_nativo.py')
best=read(run/'mejor.json')
record=dict(epochs_completed=saved['epoch'],steps=saved['steps'],best_epoch=best['epoch'],model_sha256=sha(run/'modelo.pt'),training_elapsed_seconds=saved['elapsed'],source_sha256=sha(run/'fuentes/entrenar_prefijo_nativo.py'),reason='Training fully completed. Freeze best checkpoint and replace only slow final inference. No retraining and no change of cohort, labels, particles or query inputs.',Test='sealed')
write(run/'congelacion_para_inferencia.json',record)
print(json.dumps(record))
