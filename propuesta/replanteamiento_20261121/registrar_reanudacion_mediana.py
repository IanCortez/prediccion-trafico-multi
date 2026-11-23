from base_replanteo import *
run=OUT/'mediana_directa_s0';r=read(run/'recuperacion_memoria_20261122/resultado.json')
assert sha(OUT/'entrenar_mediana_directa_recuperar.py')==r['recovery_runner_sha256']==sha(run/'fuentes/entrenar_mediana_directa_recuperar.py')
h=read(run/'historia.json');assert len(h)>=2 and h[1]['epoch']==2
s=read(OUT/'CONTINUIDAD_OPERATIVA.json')
s['current_trial'].update(pid=34040,exec_session_id=47742,serial_runner_pid=None,
 stage='Recovered epoch2 completed; epochs3..6 jointly train explicit point readout and student. One GPU. Original sources preserved.',
 runner='entrenar_mediana_directa_recuperar.py',recovery_source_sha256=r['recovery_runner_sha256'])
s['next_after_result'][0]='Observe actual directmedian recoveryPID34040/session47742 and mediana_directa_recuperar_s0.log; epoch2 completed after source-preserving memory-only fix.'
s['next_after_result'][2]='At terminal: audit every epoch. Certify any learned selected best; use SAME entrenar_mediana_directa_recuperar.py --mode frozen_control for conditional control, not original memory-retaining runner.'
write(OUT/'CONTINUIDAD_OPERATIVA.json',s)
print(json.dumps(dict(recovery_live_epoch=h[-1]['epoch'],epoch1_preserved=True,sources_frozen=True)))
