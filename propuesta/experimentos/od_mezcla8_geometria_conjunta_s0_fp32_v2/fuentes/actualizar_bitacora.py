from common import *
from evaluar_iteracion import *
import matplotlib.pyplot as plt
records=read_records();records.sort(key=lambda r:r['validation']['mae_s']);primary_all=[r for r in records if eligible(r)]
families={}
for r in primary_all:
    key=r.get('trained_model_source',r['id'])
    if key not in families or r['config'].get('inference_precision')=='float32':families[key]=r
primary=sorted(families.values(),key=lambda r:r['validation']['mae_s'])
evaluations={r['id']:report_iteration(r) for r in records}
comparisons=[]
if primary:
    best=primary[0];b=np.load(RUNS/best['id']/'validacion.npz');_,inv=np.unique(b['timestamp']//86400,return_inverse=True)
    counts=np.bincount(inv);rng=np.random.default_rng(1729);ix=rng.integers(0,len(counts),(4000,len(counts)))
    for r in primary[1:]:
        z=np.load(RUNS/r['id']/'validacion.npz');assert np.array_equal(z['row'],b['row'])
        delta=np.abs(b['pred']-b['true'])-np.abs(z['pred']-z['true']);sums=np.bincount(inv,weights=delta);boot=sums[ix].sum(1)/counts[ix].sum(1)
        comparisons.append(dict(best=best['id'],reference=r['id'],delta_mae_s=float(delta.mean()),ci95_s=np.quantile(boot,[.025,.975]).tolist(),caveat='Exploratory validation after adaptive selection; conditional on fitted models.'))
write_json(BASE/'instrumentos/comparaciones_validacion.json',comparisons)
lines=['# Bitácora crítica','','Todos los MAE de esta bitácora son de VALIDACIÓN. La prueba permanece reservada. Meta principal: <73,5 s.','','Cohorte primaria: 1.023.529 viajes train y 127.941 validación. CALL_TYPE y la ruta real observada se identifican como variantes no elegibles para la selección OD estricta.','','| Corrida | n train | MAE val. (s) | Elegible OD primaria |','|---|---:|---:|:---:|']
for r in records:lines.append(f"| {r['id']} | {r.get('train_n','?')} | {r['validation']['mae_s']:.3f} | {'sí' if eligible(r) else 'no'} |")
for r in records:
    ev=evaluations[r['id']];lines.extend(['','## '+r['id'],'',ev['note']])
    if 'comparison' in ev:
        q=ev['comparison'];lo,hi=q['ci95_s'];lines.extend(['',f"Diferencia frente a `{q['reference']}`: {q['delta_mae_s']:+.3f} s; IC95% pareado por día [{lo:+.3f}, {hi:+.3f}]. Un valor negativo favorece esta corrida. Es una comparación exploratoria tras selección adaptativa, condicionada a estos modelos ajustados."])
    lines.extend(['',ev['decision']])
if primary:
    winner=primary[0];base=next((r for r in primary if r['id']=='original_completo_pace_pos_s0'),None)
    lines.extend(['','## Estado de la decisión','',f"Mejor modelo individual terminado: `{winner['id']}`, MAE de validación {winner['validation']['mae_s']:.3f} s. La distancia a la meta sigue siendo {winner['validation']['mae_s']-73.5:.3f} s."])
    if base:lines.extend(['',f"El original completo corregido obtuvo {base['validation']['mae_s']:.3f} s. La mejora puntual del mejor modelo es {base['validation']['mae_s']-winner['validation']['mae_s']:.3f} s; no se extrapola a una superioridad sobre TransferTraj."])
lines.extend(['','Las siguientes decisiones se basan en las ablaciones recientes: comprobar variación entre semillas, ensamble simple y geometría de rutas calculadas. No cambiar la cohorte ni consultar la prueba para perseguir 73,5 s.','','## Comparabilidad','','TransferTraj publica 323.481 viajes de Porto; aquí se conservan 1.279.412 con filtros explícitos. El suplemento oficial no aporta esa cohorte de Porto. Hay 126.950 viajes confirmatorios no expuestos en la auditoría anterior. Una cifra en esta cohorte no replica su tabla. Solo se dispone de trayectorias completas de Porto; mapas de otras ciudades y demos públicas de 100 viajes no acreditan transferencia entre ciudades.','','La incertidumbre y los intervalos aquí no corrigen la selección adaptativa realizada sobre validación. La confirmación independiente requiere el candidato congelado y una única evaluación del bloque reservado.'])
(BASE/'BITACORA.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if primary:
    fig,ax=plt.subplots(figsize=(11,max(5,len(primary)*.32+1.5)))
    values=[r['validation']['mae_s'] for r in primary];labels=[r['id'].replace('od_','') for r in primary]
    ax.barh(labels,values,color=['#b17621']+['#376c9b']*(len(primary)-1));ax.axvline(73.5,color='#b34b3c',ls='--',label='Meta 73,5 s')
    for i,value in enumerate(values):ax.text(value+.7,i,f'{value:.2f}',va='center',fontsize=8)
    ax.set_xlim(0,max(values)+13);ax.set_xlabel('MAE de validación (segundos)');ax.invert_yaxis();ax.legend();ax.set_title('Porto, cohorte primaria OD: prueba reservada')
    fig.tight_layout();fig.savefig(BASE/'instrumentos/iteraciones.png',dpi=160);plt.close(fig)
print([(r['id'],round(r['validation']['mae_s'],3)) for r in records],flush=True)

