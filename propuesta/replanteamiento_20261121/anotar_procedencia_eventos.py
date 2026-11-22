from base_replanteo import *
import ast
folder=OUT/'prior_eventos'
r=read(folder/'resultado.json')
r['inputs']={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'datos_semi_markov/runs_train.npz',OUT/'datos_semi_markov/protocolo.json',DATA/'train_x.npz',DATA/'manifiesto.json']}
r['primary_cohort_and_labels_unchanged']=True
write(folder/'resultado.json',r)
p=OUT/'piloto_historia_condicional.py'
assert not (OUT/'piloto_historia_condicional_s0').exists()
s=p.read_text(encoding='utf-8-sig')
a="parent_sha=sha(parent/'modelo.pt')"
b=a+"\nhistory_counts_sha=sha(OUT/'prior_eventos/conteos_fold.npz')\nassert history_counts_sha==read(OUT/'prior_eventos/resultado.json')['outputs']['conteos_fold.npz']['sha256']"
assert a in s;s=s.replace(a,b)
a="cfg=dict(parent_sha256=parent_sha,"
b="cfg=dict(parent_sha256=parent_sha,history_counts_sha256=history_counts_sha,history_code_sha256=sha(OUT/'historia_eventos.py'),"
assert a in s;s=s.replace(a,b)
ast.parse(s);p.write_text(s,encoding='utf-8')
c=read(OUT/'CONTINUIDAD_OPERATIVA.json')
c['goal_turn_classification']='progress: controlled Train likelihood evidence, compact assets, exact fold-exclusion verification and conditional CPU/GPU continuation implemented; Native GPU process verified live'
c['current_trial']['session']=9112
c['active_history_supervisor']=dict(pid=35436,session=46174,script='continuar_tras_prefijo_nativo.py',state='pipeline_historia_estado.json')
c['completed_CPU_reference_check']=read(OUT/'verificacion_normalizacion_referencia_cpu.json')
c['do_not_rerun_one_time_preparation']=['preparar_historia_eventos.py: already recovered; output folder exists','preparar_entrenador_historia.py: target file already exists','condicionar_entrenador_historia.py: already applied','anotar_procedencia_eventos.py: one-time pilot edit, do not repeat']
write(OUT/'CONTINUIDAD_OPERATIVA.json',c)
print('Procedencia y continuidad verificables; el objetivo sigue activo.')
