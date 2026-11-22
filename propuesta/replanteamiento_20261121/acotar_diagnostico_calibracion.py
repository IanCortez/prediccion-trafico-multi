from base_replanteo import *
p=OUT/'calibrar_fases_una_vez.py';s=p.read_text(encoding='utf-8-sig')
s=s.replace("lo=np.minimum(pred[0],pred[-1]);hi=np.maximum(pred[0],pred[-1]);lower=np.maximum(lo-y,0)+np.maximum(y-hi,0)","oracle_tested=np.abs(pred.astype(np.float64)-y.astype(np.float64)[None,:]).min(0)")
s=s.replace("ORACLE_lower_bound_for_only_THIS_FROZEN_convex_CDF_pair_s=float(lower.astype(float).mean())","ORACLE_min_error_among_only_the8_tested_outputs_s=float(oracle_tested.mean())")
s=s.replace("oracle_scope='Privileged per-query weight chooses with unknown truth. Applies only to these two frozen distributions and this support weighting; not a general lower bound. Do not keep optimizing weights if this bound exceeds73.5s.'","oracle_scope='Privileged per-query choice among only the8 actual predeclared outputs, using unknown truth. Not deployable, not achieved MAE, and not a lower bound for untested weights or other architectures. The single-scan limit comes from the predeclared protocol.'")
s=s.replace("'ORACLE_lower_bound_for_only_THIS_FROZEN_convex_CDF_pair_s'","'ORACLE_min_error_among_only_the8_tested_outputs_s'")
compile(s,str(p),'exec');p.write_text(s,encoding='utf-8')
plan=read(OUT/'CONTINUIDAD_OPERATIVA.json');plan['pipeline_session']=64397;plan['pipeline_supervisor_pid']=30780;plan['pipeline_state_file']='pipeline_estado.json';plan['pending_CPU_control']='piloto_entorno_compartido.py';plan['calibration_oracle_scope']='Only8predeclared actual outputs; no universal or untested-weight impossibility claim.';write(OUT/'CONTINUIDAD_OPERATIVA.json',plan)
print('Calibration diagnostic restricted to actual tested outputs; no broad numerical bound claimed.')

