from base_replanteo import *
import subprocess
p=OUT/'piloto_tipo_condicional_s0/resultado.json';r=read(p)
for key in ['paired_vs_control','paired_vs_reference']:
 r[key]['interpretation']='Exploratory Train-only day-block interval. Held rows reserve only the new heads; the frozen base already saw all Train. No Validation was accessed by this instrument.'
r['postprocess_note']='Corrected generic paired-helper wording from reused Validation to actual Train-only scope; numeric results unchanged. Original executed source remains preserved and hashed.'
write(p,r)
c=read(OUT/'CONTINUIDAD_OPERATIVA.json')
c['CPU_trial']=dict(experiment='tipo_latente_cpu_s0',session=35045,device='CPU',threads=2,sources_frozen_while_running=True,prerequisite='Train-only matched pilot favorable but base not truly held out')
c['latest_storage_snapshot']=read(OUT/'inventario_conservacion.json')
c['next_after_result']=[
 'Read full result of riesgo_total_s0 and matched semantic CPU trial; neither partial losses nor Train pilot establish target.',
 'Compare semantic type treatment against equal-capacity unsupervised head and unchanged protected reference.',
 'Any full MAE below 131.06 and any new best: preserve all weights/source/predictions; qualify only after raw inference checks.',
 'If CPU baseline differs numerically from protected GPU baseline, preserve promising head provisionally and reproduce on declared canonical GPU only after the existing GPU job finishes.',
 'Draw explicit conclusions and choose genuinely distinct next work if no target. Do not revisit rejected temporal/shared pilots or the finished native local-loss training.',
 'Test remains sealed until a strong candidate is frozen and reproducible. No cohort, labels or query-information changes.'
]
c['resource_policy']='One heavy GPU job (risk), plus bounded CPU2-thread frozen-head test; RAM-only hidden/CDF arrays about 2.6GB; no disk CDF bank.'
c['files']+=['piloto_tipo_condicional_s0/resultado.json','tipo_latente_cpu_s0/plan.json','conclusion_nativo_completa.json']
write(OUT/'CONTINUIDAD_OPERATIVA.json',c)
marker='<!-- tipo_latente_control_20261122 -->'
body="""

<!-- tipo_latente_control_20261122 -->
El instrumento semántico con representación fuerte congelada terminó: en 5.475 filas reservadas de Train, el control equivalente obtuvo MAE 110,991 s y la supervisión del tipo 110,274 s; diferencia -0,718 s, IC95 % exploratorio por día [-1,370; -0,037]. La referencia obtuvo 112,748 s en esa muestra. La NLL de duración también bajó de 3,48855 a 3,48186. El intervalo se refiere a Train, no a Validación. La representación de base ya había visto estas filas, por lo que no constituye generalización independiente.

Está en marcha un control ampliado solo en CPU: encoder de referencia congelado, tres expertos semánticos de duración mediante inclinación suave de su distribución y compuerta inferida desde la representación OD. El tipo A/B/C es exclusivamente una etiqueta de Train. Se ajustan una cabeza supervisada y un control de capacidad equivalente sin esa etiqueta, con presupuesto fijo y sin seleccionar durante entrenamiento en Validación. Al finalizar se evaluarán todos los 127.941 viajes. Las representaciones y distribuciones se mantienen solo en RAM, sin bancos en disco. Ver tipo_latente_cpu_s0/plan.json y su registro. Continúa a la vez un único trabajo GPU: riesgo_total_s0.
"""
for f in [OUT/'README.md',BASE/'RESUMEN_RESULTADOS.md']:
 txt=f.read_text(encoding='utf-8-sig')
 if marker not in txt:f.write_text(txt+body,encoding='utf-8')
print('Diagnóstico semántico y continuidad registrados')
