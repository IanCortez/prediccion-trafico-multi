from base_replanteo import *
p=OUT/'auditar_conservacion_pilotos_recientes.py'
s=p.read_text(encoding='utf-8')
old=" assert (run/'protocolo.json').exists()"
new=" plans=[p for p in [run/'protocolo.json',run/'plan.json'] if p.exists()];assert len(plans)>=1\n records[-1]['configuration_files']={p.name:sha(p) for p in plans}"
assert old in s
p.write_text(s.replace(old,new),encoding='utf-8')
