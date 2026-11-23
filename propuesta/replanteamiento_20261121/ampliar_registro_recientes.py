"""Extend reporting only; immutable experimental sources stay unchanged."""
from base_replanteo import *
p=OUT/'consolidar_auditoria_resultados.py';s=p.read_text()
marker="# New full-primary experiments: corrected road prefix, journey consistency, global diffusion."
if marker not in s:
 insert="""
# New full-primary experiments: corrected road prefix, journey consistency, global diffusion.
source=OUT/'progreso_vial_recurrente_s0/resultado_etapa_A.json'
if source.exists():
 r=read(source);z=load_archive(source.parent/'validacion_solo_raiz.npz')
 append(source.parent.name,'Progreso recurrente: solo raíz',z['pred'],r['root_diagnostic_validation'],source,kind='secondary_branch')
source=OUT/'progreso_vial_recurrente_s0/etapa_B/resultado.json'
if source.exists():
 r=read(source);z=load_archive(source.parent/'validacion.npz')
 for key,m in r['metrics'].items():
  label='Progreso recurrente: '+key
  append('progreso_vial_recurrente_s0',label,z[key],m,source,kind='selected_model' if key in ['full246_MAE','control216_MAE'] else 'diagnostic_quantile')
source=OUT/'consistencia_temporal_viaje_s0/etapa_B/resultado.json'
if source.exists():
 r=read(source);z=load_archive(source.parent/'validacion.npz')
 append('consistencia_temporal_viaje_s0','Consistencia temporal del viaje',z['pred'],r['main_validation'],source)
source=OUT/'difusion_global_tiempo_geometria_s0/resultado.json'
if source.exists():
 r=read(source);z=load_archive(source.parent/'validacion.npz')
 append(source.parent.name,'Difusión global conjunta',z['pred'],r['validation'],source)
source=OUT/'memoria_subtrayectos_cdf_s0/resultado.json'
if source.exists():
 r=read(source);z=load_archive(source.parent/'validacion.npz')
 append(source.parent.name,'Subtrayectos: mezcla según evidencia',z['pred'],r['metrics']['evidence_mixture'],source)
 append(source.parent.name,'Subtrayectos: memoria directa con respaldo',z['kernel_backoff'],r['metrics']['kernel_backoff'],source,kind='secondary_branch')
"""
 assert s.count('pending=[')==1;s=s.replace('pending=[',insert+'\npending=[');p.write_text(s)
# Reporting continuity only, actual process is recorded independently.
state=read(OUT/'CONTINUIDAD_OPERATIVA.json')
state['current_trial']=dict(experiment='memoria_subtrayectos_cdf_s0',stage='Fullprimary evaluation; frozen sourcekernel and evidencegate. Raw65/order/singleton proof included.',Test='sealed')
state['next_after_result']=['Inspectactualresult/log before rerun; do not reinterpret observationtimeouts.',
 'Use outcome to decide Train-only calibration/learned subtrajectory attention, changing source inference not auxiliary head.',
 'Preserve all predictions/metrics, best+lineage, original data and TransferTraj; no largepair orCDFbank.']
write(OUT/'CONTINUIDAD_OPERATIVA.json',state)
active=read(OUT/'arquitectura_activa.json');active.update(current_experiment='memoria_subtrayectos_cdf_s0',
 current_protocol='memoria_subtrayectos_cdf_s0/protocolo.json',
 checkpoint_kind='NonparametricTrain subtrajectorydensity plus protectedCDF backoff',
 completed_recent={'progreso_recurrente_MAE_s':136.1112052749399,'consistencia_temporal_MAE_s':135.99613162839904,
 'difusion_global_MAE_s':139.95339397347203})
write(OUT/'arquitectura_activa.json',active)
