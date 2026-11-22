from base_replanteo import *
import ast
p=OUT/'entrenar_prefijo_historia.py'
assert not (OUT/'prefijo_historia_s0').exists()
s=p.read_text(encoding='utf-8-sig')
a="assert read(OUT/'verificacion_historia_eventos.json')['passed']"
b=a+"\n    pilot=read(OUT/'piloto_historia_condicional_s0/resultado.json')\n    assert pilot['worth_GPU_test'],'Conditional pilot does not justify this GPU trial'"
assert a in s;s=s.replace(a,b)
a="missing=model.load_native_initialization(ck['state'])"
b=a+"\n    head_path=OUT/'piloto_historia_condicional_s0/cabezas.pt'\n    head_sha=sha(head_path)\n    assert head_sha==pilot['head_sha256']\n    head=torch.load(head_path,map_location='cpu',weights_only=False)\n    assert head['parent_sha256']==parent_sha and head['worth_GPU_test']\n    model.event_residual.load_state_dict(head['history_head'])\n    del head"
assert a in s;s=s.replace(a,b)
a="event_probability_sha256=event_probability_sha,event_counts_sha256=event_counts_sha,history_code_sha256"
b="event_probability_sha256=event_probability_sha,event_counts_sha256=event_counts_sha,head_initialization_sha256=head_sha,history_code_sha256"
assert a in s;s=s.replace(a,b)
s=s.replace("New3-event residual logits initialized0 around empirical prior; native stay/stop heads frozen and unused","New3-event residual head initialized from controlled CPU Train-only pilot; native stay/stop heads frozen and unused")
s=s.replace("Best generated graph MAE on fixed512 Validation queries; alpha chosen only after checkpoint frozen","Best alpha1 output with mass-dependent reference shrinkage on fixed512 Validation queries; alpha chosen only after checkpoint frozen")
ast.parse(s);p.write_text(s,encoding='utf-8')
r=read(OUT/'preparacion_entrenador_historia.json')
r.update(target_sha256=sha(p),requires_conditional_pilot_to_beat_both_native_and_matched_head=True,CPU_head_initialization=True)
write(OUT/'preparacion_entrenador_historia.json',r)
print('GPU nueva condicionada a mejora frente al modelo neuronal y su control.')
