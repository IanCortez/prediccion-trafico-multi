"""Trainoptimization instrument only, no candidate metric or checkpoint alteration."""
from base_replanteo import *
import torch
run=OUT/'atencion_subtrayectos_s0';p=run/'ultima_reanudacion.pt';state=torch.load(p,map_location='cpu',weights_only=False)
sd=state['branch_state'];record=dict(epoch=state['epoch'],train_loss_history=state['history'],
 gate_final_bias=sd['gate.2.bias'].numpy(),gate_final_weight_norm=float(sd['gate.2.weight'].norm()),
 source_head_final_weight_norm=float(sd['head.2.weight'].norm()),source_head_final_bias_norm=float(sd['head.2.bias'].norm()),
 interpretation='Optimizationdiagnostic only. Gatebiasalone isnot averageconditionalmixtureweight; do notlabel it collapse.',
 actual_model_parent_frozen=True,Validation_access=False,Test_access=False)
write(OUT/'diagnostico_ajuste_rama_subtrayectos.json',record)
print(json.dumps(read(OUT/'diagnostico_ajuste_rama_subtrayectos.json')),flush=True)
