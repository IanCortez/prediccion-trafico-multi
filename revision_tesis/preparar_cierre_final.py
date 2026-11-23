from pathlib import Path
import json,hashlib,time,re
ROOT=Path(r"E:\UTEC\Tesis\TesisIan")
OUT=ROOT/"publicacion_github_20261122"
DEV=ROOT/"propuesta/replanteamiento_20261121"
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024**2),b""):h.update(block)
    return h.hexdigest()
copied=[]
def copy(source):
    relative=source.relative_to(ROOT)
    payload=source.read_bytes()
    assert hashlib.sha256(payload).hexdigest()==sha(source),str(source)
    destination=OUT/relative;destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_bytes(payload)
    copied.append({"path":relative.as_posix(),"bytes":len(payload),"sha256":sha(destination)})
def text(relative,value):
    p=OUT/relative;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(value,encoding="utf-8")
closure=json.loads((DEV/"CIERRE_ULTIMA_PROPUESTA.json").read_text(encoding="utf-8-sig"))
assert closure["status"]=="COMPLETE_AND_CLOSED_BY_USER" and closure["target_achieved"] is False
assert sha(DEV/"RESUMEN_FINAL_ULTIMA_PROPUESTA.txt")==closure["summary_sha256"]
for source in sorted(DEV.rglob("*")):
    if not source.is_file() or any(v in source.parts for v in ("__pycache__",".git")):continue
    if source.suffix.lower() in (".py",".json",".md",".txt",".csv",".png",".svg"):
        if source.stat().st_size<10*1024**2:copy(source)
    elif source.suffix.lower()==".npz" and source.name.startswith(("validacion","predicciones","reproduccion")) and "progreso" not in source.name:
        if source.stat().st_size<95*1024**2:copy(source)
for name in ("ARQUITECTURA.md","RESUMEN_RESULTADOS.md","README.md"):
    copy(ROOT/"propuesta"/name)
last=DEV/"red_cinetica_Train_s0"
record=json.loads((last/"resultado.json").read_text(encoding="utf-8-sig"))
for entry in record["weights"].values():
    source=Path(entry["file"]);assert sha(source)==entry["sha256"]
    copy(source)
copy(last/"constructor.pt")
ref=json.loads((DEV/"referencia_publica_Train_red_vial.json").read_text(encoding="utf-8-sig"))
reference=Path(ref["model_file"]);assert sha(reference)==ref["model_sha256"];copy(reference)
for source in reference.parent.glob("constructor*"):
    if source.is_file() and source.stat().st_size<2*1024**2:copy(source)
text("revision_tesis/CIERRE_FINAL.md","""# Cierre final para revisión de la tesis

Cierre del 23 de noviembre de 2026, solicitado después de terminar la última propuesta.
**La investigación experimental quedó cerrada. No se alcanzó MAE <73,5 s. Test permanece
sin evaluar.** El registro final indica que no quedan experimentos activos.

## Mejor resultado principal

**129,990034 s en los mismos 127.941 viajes de Validation**, del modelo de atención sobre
subtrayectos históricos. Se mantienen su checkpoint, los dos padres requeridos, predicciones,
fuentes y evidencia de reproducción completa. La mejora frente a su padre inmediato fue
0,168007 s: pequeña, con una sola semilla y selección adaptativa; no cumplió el criterio
práctico de 0,5 s. No es un resultado de confirmación independiente.

El ensamble histórico de 131,063381 s sigue documentado con sus componentes incompletos.
El modelo original de 131,256712 s y los archivos de predicciones TransferTraj se conservan.

## Última comparación: resultado negativo

Las tres cifras siguientes corresponden a **10.000 consultas cronológicas reservadas dentro
de Train**. No deben compararse directamente con los 129,99 s de Validation primaria.

| Variante | MAE (s) | RMSE (s) |
|---|---:|---:|
| Referencia pública congelada | **129,6225** | 204,4502 |
| Red vial con posición y compuerta | 132,6075 | 207,6219 |
| Red vial con movimiento y compuerta | 131,9850 | 206,7393 |

La red con movimiento empeoró 2,3625 s frente a la referencia, IC95% exploratorio por día
[1,4465; 3,3752]. Frente a la red de posición mejoró 0,6225 s, pero el intervalo
[-1,5102; 0,2410] incluye cero. Ninguna variante pasó los criterios predeclarados:
5 s frente a la referencia y, para movimiento, 2 s adicionales frente al control de posición.
No se amplió esta familia a Validation primaria ni se hicieron nuevas búsquedas después del resultado.

La red aislada, antes de la compuerta, obtuvo 280,248 s para posición y 268,386 s para movimiento.
La cobertura de red del 99,24 % no se tradujo en precisión. Las compuertas dieron un peso medio
cercano al 2 % a la red, pero aun así empeoraron la referencia. Una explicación plausible es que
condicionar los extremos no sustituye una política de rutas específica del destino; no se aisló
esa causa experimentalmente. También se documenta la asimetría de ajuste entre referencia y red.

## Qué puede sostenerse

- La mejor propuesta de atención aporta una mejora numérica pequeña en validación reutilizada.
- Modelar calles, movimiento, geometría o esperas no garantizó mejorar el tiempo total OD.
- Las pruebas de integridad y reproducción pasaron; su éxito no demuestra utilidad predictiva.
- Los pilotos de Train, los diagnósticos con información privilegiada y las cotas de familias
  fijas no acreditan el objetivo ni un límite del error irreducible.
- Los 73,5 s de TransferTraj proceden de otra cohorte procesada de Porto no disponible aquí.
- Faltan confirmación independiente y repetición entre semillas para establecer generalización.

## Evidencia y revisión

- [Informe final íntegro](../propuesta/replanteamiento_20261121/RESUMEN_FINAL_ULTIMA_PROPUESTA.txt).
- [Registro formal de cierre](../propuesta/replanteamiento_20261121/CIERRE_ULTIMA_PROPUESTA.json).
- [Resultado de la última comparación](../propuesta/replanteamiento_20261121/red_cinetica_Train_s0/resultado.json).
- [Conclusiones completas y limitaciones](../propuesta/replanteamiento_20261121/red_cinetica_Train_s0/conclusion.json).
- [Reproducción independiente registrada](../propuesta/replanteamiento_20261121/red_cinetica_Train_s0/reproduccion_independiente.json).
- [Mejor modelo y sus dependencias](ACTUALIZACION_129_99.md).

Desde la raíz de la rama:

    python -B revision_tesis/verificar_resultados.py
    python -B revision_tesis/verificar_nuevo_modelo.py
    python -B revision_tesis/verificar_cierre.py

El último comando recalcula los MAE de la comparación final, valida alineación, distribuciones
guardadas y huellas de los pesos. No reentrena, no ejecuta GPU, no abre Test ni sustituye la
reproducción de inferencia independiente ya registrada. Los datos grandes para inferencia
completa siguen siendo recursos externos, igual que en las entregas previas.

Se incluyen los dos pesos finales de compuerta de esta comparación, su constructor y el peso
de la referencia pública congelada. Sus resultados son de Train reservado y no sustituyen al
mejor modelo primario. Las copias originales de la investigación permanecen intactas.
""")
text("README.md","""# Predicción de tráfico: resultados y cierre de la investigación

**Cierre final: 23 de noviembre de 2026. Meta MAE <73,5 s no alcanzada. Test sin evaluar.**
La última propuesta terminó y la investigación se cerró por indicación del usuario.

**Mejor MAE completo verificado: 129,990034 s sobre 127.941 viajes de Validation.**
Se conservan el nuevo checkpoint y sus dos padres, el modelo original, las predicciones,
fuentes, protocolos y evidencia de reproducción.

- [Comenzar por el cierre final y sus conclusiones](revision_tesis/CIERRE_FINAL.md).
- [Informe final íntegro](propuesta/replanteamiento_20261121/RESUMEN_FINAL_ULTIMA_PROPUESTA.txt).
- [Guía de revisión para Ian](revision_tesis/README.md).
- [Modelo de 129,99 s: cadena de dependencias y limitaciones](revision_tesis/ACTUALIZACION_129_99.md).
- [Historial de resultados](propuesta/RESUMEN_RESULTADOS.md).
- [Arquitectura y límites científicos](propuesta/ARQUITECTURA.md).
- [Comparaciones locales con TransferTraj](propuesta/comparaciones_transfertraj/README.md).

La última red vial no superó su referencia en 10.000 consultas reservadas de Train:
131,9850 s frente a 129,6225 s. Son datos distintos de Validation primaria.
La reproducción pasó, pero no hubo una mejora predictiva útil.

El código original del alumno en src/ permanece intacto. Las memorias históricas grandes y
los mapas siguen siendo recursos externos identificados por SHA256. Los registros antiguos
de procesos son históricos; el documento CIERRE_ULTIMA_PROPUESTA.json fija el estado final.
""")
guide=OUT/"revision_tesis/README.md"
value=guide.read_text(encoding="utf-8")
banner="""## Estado final del 23 de noviembre

**Investigación cerrada. Mejor MAE de Validation: 129,990034 s. Meta de 73,5 s no alcanzada;
Test sin evaluar.** Leer primero [CIERRE_FINAL.md](CIERRE_FINAL.md) para distinguir el resultado
principal de los pilotos y conocer las conclusiones negativas de la última red vial.
Los apartados siguientes conservan la guía de las entregas de desarrollo.

"""
if "## Estado final del 23 de noviembre" not in value:
    value=value.replace("# Guía para Ian: revisión de resultados y modelos\n",
                        "# Guía para Ian: revisión de resultados y modelos\n\n"+banner,1)
guide.write_text(value,encoding="utf-8")
for name in ("README.md","ARQUITECTURA.md","RESUMEN_RESULTADOS.md"):
    p=OUT/"propuesta"/name
    value=p.read_text(encoding="utf-8-sig")
    notice="""> **Cierre del 23 de noviembre:** la última propuesta terminó y la investigación quedó cerrada.
> Mejor MAE primario: 129,990034 s; meta de 73,5 s no alcanzada; Test sin evaluar.
> Ver [cierre final](../revision_tesis/CIERRE_FINAL.md). Los planes y estados siguientes son históricos.

"""
    p.write_text(notice+value,encoding="utf-8")
checker=OUT/"revision_tesis/verificar_resultados.py"
code=checker.read_text(encoding="utf-8")
code=code.replace('if "validation" not in record or record["validation"].get("n")!=127941:continue',
'''validation=record.get("validation")
            if not isinstance(validation,dict) or validation.get("n")!=127941:continue''')
checker.write_text(code,encoding="utf-8")
text("revision_tesis/verificar_cierre.py",'''"""Recalculate saved final Train-pilot metrics without training, GPU inference or Test access."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DEV=ROOT/"propuesta/replanteamiento_20261121"
def read(path):return json.loads(path.read_text(encoding="utf-8-sig"))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    closure=read(DEV/"CIERRE_ULTIMA_PROPUESTA.json")
    assert closure["status"]=="COMPLETE_AND_CLOSED_BY_USER"
    assert closure["target_achieved"] is False and closure["Test"]=="SEALED"
    assert sha(DEV/"RESUMEN_FINAL_ULTIMA_PROPUESTA.txt")==closure["summary_sha256"]
    run=DEV/"red_cinetica_Train_s0"
    record=read(run/"resultado.json")
    assert record["complete"] and record["Validation"] is False and record["Test"] is False
    with np.load(run/"predicciones_held_Train.npz",allow_pickle=False) as data:
        saved={k:data[k] for k in data.files}
    assert len(saved["row"])==len(np.unique(saved["row"]))==10000
    assert sha(run/"predicciones_held_Train.npz")==record["predictions_sha256"]
    computed={}
    for arm in ("position","kinetic","reference"):
        pred=saved[arm].astype(float);true=saved["true"].astype(float)
        assert pred.shape==true.shape==(10000,) and np.isfinite(pred).all()
        value=float(np.abs(pred-true).mean())
        assert abs(value-record["metrics"][arm]["mae_s"])<1e-10
        computed[arm]=value
        if arm=="reference":continue
        with np.load(run/arm/"predicciones_held_Train.npz",allow_pickle=False) as data:
            assert np.array_equal(data["row"],saved["row"])
            assert np.array_equal(data["true"],saved["true"])
            assert np.array_equal(data["pred"],saved[arm])
            assert np.array_equal(data["quantiles"][:,1],saved[arm])
            prob=np.exp(data["log_prob"])
            assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-10,rtol=0)
            assert int((~data["supported"]).sum())==76
            assert np.array_equal(data["pred"][~data["supported"]],saved["reference"][~data["supported"]])
        weights=record["weights"][arm]
        assert sha(run/arm/"epoca_80.pt")==weights["sha256"]
    ref=read(DEV/"referencia_publica_Train_red_vial.json")
    assert sha(DEV/ref["id"]/"epoca_4.pt")==ref["model_sha256"]
    proof=read(run/"reproduccion_independiente.json")
    assert proof["passed"] and proof["reference_all10000_exact"]
    for arm in ("position","kinetic"):
        assert proof["arms"][arm]["passed"] and proof["arms"][arm]["all10000_quantiles_exact"]
    assert computed["kinetic"]>computed["reference"]
    assert record["hypothesis_position_passed"] is False and record["hypothesis_kinetic_passed"] is False
    print(json.dumps({"passed":True,"closed":True,"scope":"10000 reused held-Train queries, not primary Validation",
                      "MAE_s":computed,"best_primary_Validation_MAE_s":129.99003446901307,
                      "target_achieved":False,"Test_read":False,"GPU_used":False,"training_run":False},indent=2))
if __name__=="__main__":main()
''')
# Refresh the external-resource note without copying its large inference datasets.
external_note={
 "snapshot_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
 "closed":True,"target_achieved":False,"Test_evaluated":False,
 "copy_only_original_research_unchanged":True,"copied_files":copied,
 "last_comparison_weights_and_reference_included":True,
 "last_comparison_large_counts_and_index_arrays_external":True,
 "primary_best_unchanged_s":129.99003446901307
}
text("revision_tesis/procedencia_cierre_final.json",json.dumps(external_note,ensure_ascii=False,indent=2))
patterns=[re.compile(r"gh[pousr]_[A-Za-z0-9]{25,}"),re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
          re.compile(r"hf_[A-Za-z0-9]{30,}"),re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{35,}"),
          re.compile(r"AKIA[0-9A-Z]{16}"),re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")]
for item in copied:
    source=OUT/item["path"]
    if source.suffix.lower() in (".py",".json",".md",".txt",".csv"):
        value=source.read_text(encoding="utf-8-sig",errors="replace")
        assert not any(p.search(value) for p in patterns),item["path"]
files=[]
for p in sorted(OUT.rglob("*")):
    if not p.is_file() or ".git" in p.parts:continue
    rel=p.relative_to(OUT)
    if rel.as_posix()=="revision_tesis/manifiesto_archivos.json":continue
    if rel.parts[0] not in ("propuesta","auditoria","revision_tesis") and rel.name not in ("README.md","requirements-revision.txt",".gitignore",".gitattributes"):continue
    assert p.stat().st_size<95*1024**2,(rel,p.stat().st_size)
    files.append({"path":rel.as_posix(),"bytes":p.stat().st_size,"sha256":sha(p)})
text("revision_tesis/manifiesto_archivos.json",json.dumps({
 "files":files,"file_count":len(files),"total_bytes":sum(x["bytes"] for x in files),
 "raw_Train_Test_and_environment_excluded":True,"secret_literal_scan_passed":True,
 "original_repository_files_not_rehashed_as_research_artifacts":True
},ensure_ascii=False,indent=2))
print(json.dumps({"files":len(files),"MiB":sum(x["bytes"] for x in files)/1024**2,"closure":closure["status"]}))
