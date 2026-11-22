from common import *
name='od_mezcla_proyectada_mm_calendario_s0_fp32_v2';z=dict(np.load(RUNS/name/'validacion.npz'));p=dict(np.load(BASE/'instrumentos/diagnostico_pausas_validation.npz'));g=dict(np.load(DATA/'geometria_validation.npz'));d=dict(np.load(DATA/'validation_x.npz'));assert np.array_equal(z['row'],p['row']) and np.array_equal(z['row'],g['row']) and np.array_equal(z['row'],d['row']);y=z['true'].astype(float);error=np.abs(y-z['pred']);stop=p['low_displacement_seconds'][:,1];ratio=g['X'][:,0]/d['od_m'];od_speed=d['od_m']/y*3.6
out={}
for key,values,bounds in [('low_displacement5m_seconds',stop,[0,.1,30,60,180,360,2000]),('actual_route_circuity',ratio,[0,1.3,2,3,5,10000]),('od_speed_kmh',od_speed,[0,2,5,10,20,40,1000])]:
    rows=[]
    for low,high in zip(bounds[:-1],bounds[1:]):
        mask=(values>=low)&(values<high)
        if mask.any():rows.append(dict(interval=[low,high],n=int(mask.sum()),fraction=float(mask.mean()),mae_s=float(error[mask].mean()),contribution_to_global_mae_s=float(error[mask].sum()/len(y)),error_share=float(error[mask].sum()/error.sum())))
    out[key]=rows
quality=[]
for label,mask in [('max_GPS_step_le500m',p['reliable']),('max_GPS_step_gt500m',~p['reliable'])]:quality.append(dict(group=label,n=int(mask.sum()),fraction=float(mask.mean()),mae_s=float(error[mask].mean()),error_share=float(error[mask].sum()/error.sum())))
rank=np.sort(error)[::-1];tail={str(c):dict(n=int(np.ceil(c*len(y))),error_share=float(rank[:int(np.ceil(c*len(y)))].sum()/error.sum())) for c in (.01,.05,.1)};record=dict(id=name,validation_mae_s=float(error.mean()),strata=out,gps_step_diagnostic=quality,largest_error_shares=tail,global_cohort_unchanged=True,test_labels_read=False,privileged_diagnostics=True,caveat='Observed low displacement, path circuity and OD speed use validation truth for diagnosis only. They are never query inputs or filters for the primary goal. Low displacement can include slow motion/GPS noise; it is not annotated parking time. Association does not prove causation.');write_json(BASE/'instrumentos/diagnostico_error_pausas.json',record);print(json.dumps(record),flush=True)
