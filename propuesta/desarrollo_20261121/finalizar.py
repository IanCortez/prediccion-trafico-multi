"""Summarize completed trials and propose a strictly scoped cleanup; never evaluates Test."""
from pathlib import Path
import json,hashlib,numpy as np
ROOT=Path('E:/UTEC/Tesis/TesisIan');BASE=ROOT/'propuesta';OUT=BASE/'desarrollo_20261121';BEST=BASE/'experimentos/od_mezcla8_geometria_conjunta_s0_fp32_v2'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 old=load(BEST/'resultado.json');fresh=load(OUT/'modelo_global_conjunto_s0/resultado.json');regional=load(OUT/'piloto_transferencia/resultado.json');online=load(OUT/'estado_causal/resultado.json');fine=load(OUT/'mae_mediana_s0/resultado.json');adapters=load(OUT/'adaptacion_rutas/resultado.json');baseline=old['validation']['mae_s'];score=fresh['validation']['mae_s'];assert min(score,baseline)>=73.5,'A candidate met Validation target; confirmation workflow required, do not close.'
 rows=[dict(experiment='Referencia conservada',protocol='Primario OD estatico',mae_s=baseline,n=127941),dict(experiment='Ajuste directo MAE: mejor epoca entrenada',protocol='Primario OD estatico',mae_s=min(r['validation']['mae_s'] for r in load(OUT/'mae_mediana_s0/historia.json')),n=127941)]
 for name in ['control_capacidad','rutas_globales']:
  h=load(OUT/'adaptacion_rutas'/name/'historia.json');rows.append(dict(experiment='Adaptador '+name+': mejor epoca entrenada',protocol='Primario OD estatico',mae_s=min(r['validation']['mae_s'] for r in h),n=127941))
 rows.append(dict(experiment='Modelo conjunto nuevo desde cero',protocol='Primario OD estatico',mae_s=score,n=127941))
 rows.extend([dict(experiment='Regional objetivo solo',protocol='Pocas etiquetas, solo zona este',mae_s=regional['metrics']['objetivo_solo']['mae_s'],n=49434),dict(experiment='Regional transferencia selectiva',protocol='Pocas etiquetas, solo zona este',mae_s=regional['metrics']['fuentes_selectivas']['mae_s'],n=49434),dict(experiment='Estado causal local 2h',protocol='Online con etiquetas de viajes ya completados; diferente del estatico',mae_s=online['metrics']['local_2h']['mae_s'],n=127941)])
 summary=dict(fecha='2026-11-21',objective_mae_s=73.5,objective_achieved=False,Test_evaluated=False,primary_best_mae_s=min(baseline,score),primary_best_model=str((OUT/'modelo_global_conjunto_s0' if score<baseline else BEST).relative_to(ROOT)),remaining_gap_s=min(baseline,score)-73.5,results=rows,fresh_model=dict(best_epoch=fresh['best_epoch'],epochs_run=fresh['epochs_run'],paired=fresh['paired_vs_retained'],seconds=fresh['seconds']),scientific_limits=['Validation reutilizada de forma adaptativa; intervalos exploratorios.', 'El piloto regional y la simulacion online no certifican el objetivo del benchmark estatico.', 'La cohorte publicada de TransferTraj no esta disponible; 73.5 s es referencia externa.', 'Estos resultados no prueban que el objetivo sea matematicamente imposible.'],preserved_evidence=load(OUT/'verificacion_conservados.json'))
 write(OUT/'resumen_resultados.json',summary)
 files=[]
 if score>=baseline:files=[OUT/'modelo_global_conjunto_s0/modelo.pt',OUT/'rutas_globales_completas/train_rutas_globales.npz',OUT/'rutas_globales_completas/validation_rutas_globales.npz']
 saved=np.load(BEST/'validacion.npz')
 for variant in ['control_capacidad','rutas_globales']:
  p=OUT/'adaptacion_rutas'/variant/'validacion.npz';a=np.load(p)
  if all(np.array_equal(a[k],saved[k]) for k in ['row','timestamp','true','pred']):files.append(p)
 plan=[]
 for p in files:
  p=p.resolve();assert p.is_relative_to(OUT.resolve()) and p.is_file();plan.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),reason='Discarded intermediate/model, or point predictions identical to retained baseline. Metrics, code and distinct predictions remain.'))
 write(OUT/'limpieza_plan.json',plan)
 table='\n'.join('| '+r['experiment']+' | '+r['protocol']+' | '+str(r['n'])+' | '+f"{r['mae_s']:.6f}"+' |' for r in rows)
 text=f'''# Resultado operativo del 21 de noviembre de 2026

El objetivo MAE <73,5 s no se alcanzo. La prueba final permanece cerrada.
Mejor MAE del protocolo primario: {min(baseline,score):.6f} s. Brecha pendiente: {min(baseline,score)-73.5:.6f} s.

| Experimento | Protocolo | Viajes | MAE (s) |
|---|---|---:|---:|
{table}

Los adaptadores y el ajuste directo usan el modelo original como retorno: su resultado seleccionado coincide con el original cuando ninguna epoca lo mejora. La tabla muestra el mejor resultado realmente entrenado, evitando atribuirles el resultado del retorno.

El modelo conjunto nuevo se entreno desde cero con 237 caracteristicas, ocho componentes de duracion y supervision geometrica auxiliar solo de entrenamiento. Agrega una ruta global con costes GPS historicos y preferencia por enlaces respaldados por entrenamiento. Se excluye el pliegue semanal de la consulta durante entrenamiento. Conserva todas las consultas de validacion, incluso si falla el calculo de ruta; en ese caso usa una representacion de respaldo. Selecciono la epoca {fresh['best_epoch']} de {fresh['epochs_run']}; el resultado final es float32 con lote canonico 2048.

La prueba geometrica inicial mejora 6,6% al control mas simple. Frente a las rutas que ya recibia el modelo, la diferencia es pequena y su intervalo incluye cero. Mejor geometria no garantiza mejor tiempo. No se afirma haber implementado el entrenamiento neuronal decision-focused del articulo de referencia.

El estado causal admite solo viajes ya completados, con una demora de publicacion de 60 segundos; supera una prueba de invariancia a etiquetas futuras. Es un protocolo diferente y su 130,42 s no certifica la meta original.

Las comparaciones usan validacion previamente reutilizada: son exploratorias. No se cambiaron filtros, unidades, etiquetas ni cobertura para alcanzar la cifra. No se obtuvo la cohorte publicada de TransferTraj. Estos resultados no establecen una cota de imposibilidad.

Se conservan codigo, configuraciones, historias, metricas, diagnosticos y predicciones distintas. Los intermediarios descartados y las predicciones identicas se identifican en limpieza_plan.json y limpieza_resultado.json. El modelo original y las diez predicciones TransferTraj conservan sus SHA256.

El entorno sigue siendo el Conda local documentado en ../../auditoria. El modelo principal conservado se reproduce con ../reproducir_mejor.py segun ../REPRODUCIR.md. Los scripts de esta carpeta registran sus configuraciones y rechazan sobreescribir una corrida terminada. Para repetir un experimento, debe asignarse una carpeta de salida nueva dentro del proyecto; los recursos originales y las fuentes congeladas siguen intactos.
'''
 (OUT/'README.md').write_text(text,encoding='utf-8');print(json.dumps({'primary_best_mae_s':summary['primary_best_mae_s'],'fresh_mae_s':score,'planned_removal_bytes':sum(r['bytes'] for r in plan),'Test_evaluated':False}))
if __name__=='__main__':main()

