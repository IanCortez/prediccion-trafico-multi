from base_replanteo import *
s=read(OUT/'sesiones_extra.json')
write(OUT/'sesiones_activas.json',dict(status='active research goal',active=s['active'],pending=['Read full result, run the single predeclared calibration, preserve qualifying candidates, then choose next mechanism from controlled evidence'],max_heavy_GPU_jobs=1,final_goal_s=73.5,preserve_candidate_threshold_s=131.06,Test='sealed',last_update_unix=time.time()))
p=OUT/'calibrar_fases_una_vez.py';code=p.read_text(encoding='utf-8-sig');code=code.replace("torch.cuda.set_per_process_memory_fraction(.25,0)\n","");code=code.replace("assert (parent/'resultado.json').exists(),'Do not run while training is active'","assert (parent/'resultado.json').exists(),'Do not run while training is active'\ntorch.cuda.set_per_process_memory_fraction(.25,0)");compile(code,str(p),'exec');p.write_text(code,encoding='utf-8')
progress=read(OUT/'espacial_fases_referencia_s0/mejor.json');state=read(OUT/'CONTINUIDAD_OPERATIVA.json');state['latest_selection_checkpoint']=progress;state['goal_is_not_complete']=True;state['active_session']=73257;state['do_not_modify_running_sources']=True;write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
print(json.dumps(dict(active=s['active'],latest_selection=progress)))

