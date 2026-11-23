from base_replanteo import *
old=OUT/'experto_subtrayectos.py';p=OUT/'experto_subtrayectos_cdf.py'
s=old.read_text()
s=s.replace('def quantiles(prob):\n cdf=prob.cumsum(1)','def quantiles(cdf):')
s=s.replace(' direct=np.where(exists[:,None],prob,base)',' source_cdf=prob.cumsum(1)\n direct=np.where(exists[:,None],source_cdf,base)')
s=s.replace('mixture=(1-gate[:,None])*base+gate[:,None]*prob','mixture=(1-gate[:,None])*base+gate[:,None]*source_cdf')
assert not p.exists();p.write_text(s)
p=OUT/'evaluar_memoria_subtrayectos_cdf.py';s=(OUT/'evaluar_memoria_subtrayectos.py').read_text()
s=s.replace('experto_subtrayectos import','experto_subtrayectos_cdf import')
s=s.replace("memoria_subtrayectos_s0'","memoria_subtrayectos_cdf_s0'")
s=s.replace("'evaluar_memoria_subtrayectos.py'","'evaluar_memoria_subtrayectos_cdf.py'")
s=s.replace("'experto_subtrayectos.py'","'experto_subtrayectos_cdf.py'")
s=s.replace("exp().mean(1).cpu().numpy()","exp().mean(1).cumsum(-1).cpu().numpy()")
s=s.replace("# Float32 cumulative arithmetic is kept for exact reference quantile reproduction.",
 "# Compute parent CDF on its original device before transfer: numpy serial Float32 cumsum differed at threshold; no target tuning.")
assert not p.exists();p.write_text(s)
write(OUT/'memoria_subtrayectos_s0/interrupcion_control_numerico.json',dict(
 completed_scientific_output=False,stage='Firstbatchbeforecandidateevaluation',
 cause='CPU NumPy sequential Float32 cumsum differed from original GPUtorchcumsum at a discrete quantilethreshold.',
 correction='Newcdfsource uses protectedGPUCDF directly, retains frozenfirstattemptsources, same source density/gate/queries/labels.',
 rerun='memoria_subtrayectos_cdf_s0',no_target_or_hyperparameter_selection=True))
