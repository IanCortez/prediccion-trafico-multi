from common import *
from features import features
from inferencia import predict_mixture_quantiles
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
name='od_mezcla_proyectada_mm_calendario_s0_fp32_v2';d,q,w,p=predict_mixture_quantiles(RUNS/name,include_distribution=True);z=np.load(RUNS/name/'validacion.npz');assert np.array_equal(d['row'],z['row']) and np.array_equal(q[:,1],z['pred']);y=z['true'].astype(float);label=((y-60)/15).astype(int);assert np.allclose(p.sum(1),1,atol=1e-5);cdf=p.astype(float).cumsum(1);truth=np.arange(116)[None,:]>=label[:,None];crps=15*np.square(cdf[:,:-1]-truth[:,:-1]).sum(1);mass=p[np.arange(len(y)),label];before=np.where(label>0,cdf[np.arange(len(y)),np.maximum(label-1,0)],0);pit=before+np.random.default_rng(1729).random(len(y))*mass;assert pit.min()>=0 and pit.max()<=1.00001
width=q[:,2]-q[:,0];covered=(y>=q[:,0])&(y<=q[:,2]);interval_score=width+10*np.maximum(q[:,0]-y,0)+10*np.maximum(y-q[:,2],0);error=np.abs(y-q[:,1]);order=np.argsort(width,kind='stable');coverage_levels=np.linspace(.1,1,10);risk=[]
for c in coverage_levels:
    ix=order[:int(np.ceil(c*len(y)))];risk.append(dict(retained_fraction=float(c),n=len(ix),conditional_mae_s=float(error[ix].mean())))
F,_=features(d);subgroups={}
for key,values,bounds in [('od_km',d['od_m']/1000,[0,1,2,4,8,1000]),('hour_local',F[:,14],[0,6,10,16,20,24])]:
    rows=[]
    for lo,hi in zip(bounds[:-1],bounds[1:]):
        mask=(values>=lo)&(values<hi)
        if mask.any():rows.append(dict(interval=[lo,hi],n=int(mask.sum()),mae_s=float(error[mask].mean()),coverage80=float(covered[mask].mean()),mean_width80_s=float(width[mask].mean())))
    subgroups[key]=rows
hist,bins=np.histogram(pit,np.linspace(0,1,11));record=json.loads((RUNS/name/'resultado.json').read_text());out=dict(id=name,validation_mae_s=float(error.mean()),n=len(y),mean_nll=float(-np.log(np.maximum(mass,1e-30)).mean()),mean_discrete_crps_s=float(crps.mean()),coverage80=float(covered.mean()),mean_width80_s=float(width.mean()),mean_interval_score80_s=float(interval_score.mean()),randomized_pit_histogram=dict(counts=hist,bin_edges=bins,seed=1729),risk_coverage=risk,subgroups=subgroups,model_sha256=record['model_sha256'],goal_achieved=False,test_labels_read=False,caveat='Exploratory after adaptive validation. Selective error is conditional on retained queries and cannot replace global MAE. Inclusive discrete intervals have ties; randomized PIT accounts for probability mass at observed duration. No future coverage guarantee or conformal calibration claimed.');write_json(BASE/'instrumentos/calibracion_mezcla_vial.json',out);np.savez_compressed(BASE/'instrumentos/calibracion_mezcla_vial.npz',row=d['row'],pit=pit,width80_s=width,covered80=covered)
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');axes[0].plot(coverage_levels*100,[r['conditional_mae_s'] for r in risk],marker='o',color='#126782');axes[0].axhline(73.5,color='#c44536',ls='--',label='Meta global: 73,5 s');axes[0].set(xlabel='Consultas retenidas por menor incertidumbre (%)',ylabel='MAE condicionado (s)',title='Diagnóstico de selección; no cambia la meta global');axes[0].legend(frameon=False);axes[1].bar((bins[:-1]+bins[1:])/2,hist/len(y)*10,width=.085,color='#126782');axes[1].axhline(1,color='#c44536',ls='--');axes[1].set(xlabel='PIT aleatorizado de duración discreta',ylabel='Densidad',title='Calibración exploratoria de la distribución')
for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
fig.suptitle('Mezcla vial: validación, prueba reservada',fontsize=13);fig.savefig(BASE/'instrumentos/calibracion_mezcla_vial.png',dpi=180);plt.close(fig);print(json.dumps({k:out[k] for k in ('validation_mae_s','mean_nll','mean_discrete_crps_s','coverage80','mean_width80_s','risk_coverage')}),flush=True)
