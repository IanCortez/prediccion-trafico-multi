from pathlib import Path
import hashlib,json,time,re,subprocess
import numpy as np
ROOT=Path(r"E:\UTEC\Tesis\TesisIan")
OUT=ROOT/"publicacion_github_20261122"
DEV=ROOT/"propuesta/replanteamiento_20261121"
assert (OUT/".git").is_dir()
snapshot=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
def sha_bytes(value):return hashlib.sha256(value).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024**2),b""):h.update(block)
    return h.hexdigest()
copied=[]
def copy(source):
    source=Path(source)
    assert source.resolve().is_relative_to(ROOT.resolve())
    for _ in range(3):
        raw=source.read_bytes()
        if sha_bytes(raw)==sha_bytes(source.read_bytes()):break
    else:raise RuntimeError(f"Changing source during snapshot: {source}")
    relative=source.relative_to(ROOT)
    target=OUT/relative;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(raw)
    copied.append({"path":relative.as_posix(),"bytes":len(raw),"sha256":sha_bytes(raw)})
def text(relative,value):
    path=OUT/relative;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(value,encoding="utf-8")
best=json.loads((DEV/"CONTINUIDAD_OPERATIVA.json").read_text(encoding="utf-8-sig"))["retained_best"]
assert best["id"]=="atencion_subtrayectos_s0"
assert abs(best["validation"]["mae_s"]-129.99003446901307)<1e-9
for name in ("ARQUITECTURA.md","RESUMEN_RESULTADOS.md","README.md","PROTOCOLO.md","REPRODUCIR.md"):
    copy(ROOT/"propuesta"/name)
# Capture lightweight source, records and complete prediction files; no raw GPS or caches.
for folder in (DEV,ROOT/"propuesta/desarrollo_20261121"):
    for source in sorted(folder.rglob("*")):
        if not source.is_file():continue
        parts=source.relative_to(folder).parts[:-1]
        if any(p in ("__pycache__",".git","tmp") for p in parts):continue
        if source.suffix.lower() in (".py",".md",".json",".csv",".png",".svg"):
            if source.stat().st_size<10*1024**2:copy(source)
        elif source.suffix.lower()==".npz" and source.name.startswith(("validacion","predicciones")) and "progreso" not in source.name:
            if source.stat().st_size<95*1024**2:copy(source)
# Publish selected completed checkpoints, not all temporary/successive epoch states.
selected={}
for result_path in sorted(DEV.glob("*/resultado.json")):
    record=json.loads(result_path.read_text(encoding="utf-8-sig"))
    path=record.get("model_file")
    if not path or not isinstance(path,str):continue
    source=Path(path)
    if not source.is_absolute():source=ROOT/source
    if source.suffix!=".pt" or not source.is_file() or not source.resolve().is_relative_to(DEV.resolve()):continue
    if any(word in source.name for word in ("reanud","continuacion","ultima")):continue
    expected=record.get("model_sha256")
    if expected:assert sha(source)==expected,(result_path,source)
    validation=record.get("validation",{})
    if validation.get("n")!=127941:continue
    copy(source)
    selected[result_path.parent.name]={"path":source.relative_to(ROOT).as_posix(),"sha256":sha(source),
                                     "bytes":source.stat().st_size,"validation":validation}
    for constructor in result_path.parent.glob("constructor*.npz"):
        if constructor.stat().st_size<1024**2:copy(constructor)
required={
 "atencion_subtrayectos_s0/epoca_6.pt":"07db7a1d4f77647de15a1db6c8be1b68e26bdd4edf16a43aab2b0d2f905c39dd",
 "atencion_historia_individual_s0/epoca_3.pt":"6d9926e4fdc7e17f0a15dbfb6c076d3323cee5cd07fb08b16e7df56d78c9399b",
 "historica_fresca_s0/epoca_27.pt":"81c543b8a175efc11a25418def2a103dfabdc4a000379ea4b4b58e46360c31e5"
}
for name,expected in required.items():
    source=DEV/name
    assert sha(source)==expected,name
    target=OUT/source.relative_to(ROOT)
    if not target.exists():copy(source)
    assert sha(target)==expected,name
# The full point/history memories and lookup indices remain external; include their metadata/hashes.
external=[]
for relative in (
 "memoria_puntos_Train_s0/puntos_Train.npz",
 "datos_prototipos_historicos_s0/memoria_Train.npz",
 "indices_subtrayectos_s0/train_indices.npz",
 "indices_subtrayectos_s0/validation_indices.npz",
 "datos_historia_individual_s0/train_indices.npz",
 "datos_historia_individual_s0/validation_indices.npz",
 "banco_proyeccion_incierta_s0/train_banco.npz",
 "banco_proyeccion_incierta_s0/validation_banco.npz",
 "datos_prototipos_historicos_s0/train_historicos.npz",
 "datos_prototipos_historicos_s0/validation_historicos.npz"
):
    source=DEV/relative
    if source.exists():
        external.append({"path":source.relative_to(ROOT).as_posix(),"bytes":source.stat().st_size,
                         "sha256":sha(source),"included_in_Git":False})
text("revision_tesis/recursos_externos_129_99.json",json.dumps({
 "reason":"Train memories, maps and derived indices are not included as Git blobs. Model chain and small normalization files are included.",
 "full_inference_also_requires_original_model_maps_profiles_and_features":True,
 "assets":external
},ensure_ascii=False,indent=2))
text("revision_tesis/modelos_actualizados.json",json.dumps({
 "snapshot_utc":snapshot,"verified_best":best,"published_selected_checkpoints":selected,
 "best_chain_complete_in_Git":True,"complete_inference_data_in_Git":False,
 "large_assets_inventory":"recursos_externos_129_99.json","Test_evaluated":False
},ensure_ascii=False,indent=2))
text("revision_tesis/ACTUALIZACION_129_99.md","""# Actualización: mínimo verificado de 129,990034 s

Evaluación completa de los mismos 127.941 viajes de Validation. El modelo
atencion_subtrayectos_s0 terminó seis épocas fijas y su checkpoint final se recargó
para reproducir todos los cuantiles, 65 consultas desde cinco campos, orden inverso
y consulta individual. Test permanece cerrado.

| Referencia | MAE (s) |
|---|---:|
| Ensamble histórico, pesos incompletos | 131,063381 |
| Modelo original conservado | 131,256712 |
| Mezcla con proyección incierta | 131,125167 |
| Mezcla histórica entrenada desde cero | 130,204821 |
| Atención sobre viajes históricos completos | 130,158042 |
| **Atención sobre subtrayectos históricos** | **129,990034** |
| Entrenamiento conjunto desde cero | 130,685980 |
| Corrección del prior por distancia de arco | 129,994490 |

La mejora frente al padre inmediato es 0,168007 s, con intervalo exploratorio por día
[-0,222310; -0,110332] s. No pasa el criterio práctico predeclarado de 0,5 s. RMSE empeora
ligeramente. La selección adaptativa de la investigación no queda corregida por ese
intervalo. La meta de confirmación <73,5 s sigue pendiente.

## Pesos y dependencias

El nuevo archivo de 0,90 MiB contiene solo una rama adicional. **No es un modelo
independiente**. Se publican también sus dos padres:

    propuesta/replanteamiento_20261121/historica_fresca_s0/epoca_27.pt
    propuesta/replanteamiento_20261121/atencion_historia_individual_s0/epoca_3.pt
    propuesta/replanteamiento_20261121/atencion_subtrayectos_s0/epoca_6.pt

La cadena produce 38 componentes: 22 de rutas/prototipos, ocho de viajes completos
y ocho de subtrayectos. Las fuentes congeladas, normalizadores, protocolos, predicciones,
registros de protección y reproducción están incluidos. Otros checkpoints finales
conservados se enumeran en modelos_actualizados.json. Las épocas intermedias adicionales
permanecen protegidas en el proyecto local; no se duplican todas en este commit.

Para revisar la evidencia sin GPU ni memorias grandes:

    python -B revision_tesis/verificar_resultados.py
    python -B revision_tesis/verificar_nuevo_modelo.py

El segundo instrumento comprueba pesos, sus huellas, las seis/ tres/27 épocas requeridas
y el MAE completo guardado. No genera nuevas predicciones. La certificación de inferencia
completa proviene del registro de reproducción independiente que se incluye.

La inferencia desde consultas nuevas necesita, además, mapas/perfiles originales y
memorias de Train que no se suben a Git. recursos_externos_129_99.json identifica los
archivos y SHA256. El banco de puntos ocupa 253.502.723 bytes y excede el límite por archivo
de GitHub. No se presenta esta rama como una distribución autónoma de esos datos.
Los scripts históricos conservan rutas de la máquina original: deben adaptarse en una copia
sin alterar las fuentes congeladas o sus huellas. No se cambiaron las ejecuciones activas.

Los nuevos pilotos de 10.000/512 consultas Train y los diagnósticos privilegiados se
documentan por separado; no son nuevas mediciones de Validation primaria.
""")
text("README.md","""# Predicción de tráfico: avances para revisión de tesis

Rama de resultados y modelos, actualizada el 22 de noviembre de 2026.
El código original del alumno en src/ permanece intacto.

**Mejor MAE completo verificado: 129,990034 s en los mismos 127.941 viajes de Validation.**
La meta de confirmación <73,5 s sigue pendiente y Test permanece cerrado.

- [Qué cambió y cómo revisar el modelo de 129,99 s](revision_tesis/ACTUALIZACION_129_99.md).
- [Guía para Ian](revision_tesis/README.md).
- [Resultados y conclusiones de los experimentos](propuesta/RESUMEN_RESULTADOS.md).
- [Arquitectura vigente y límites científicos](propuesta/ARQUITECTURA.md).
- [Comparaciones locales con TransferTraj](propuesta/comparaciones_transfertraj/README.md).

La entrega contiene el nuevo checkpoint y sus dos padres, fuentes congeladas, métricas,
predicciones y evidencia de reproducción. Las memorias grandes de Train y mapas permanecen
como recursos externos identificados por SHA256. El ensamble histórico de 131,06 s conserva
su registro, pero tres de sus cuatro checkpoints se eliminaron antes de esta rama.
Cada actualización es una instantánea: los registros de procesos no describen ejecuciones
que sigan activas al consultar GitHub.
""")
guide=(OUT/"revision_tesis/README.md").read_text(encoding="utf-8")
guide=guide.replace("## Cifras que deben distinguirse","""## Actualización de esta entrega

El nuevo mínimo completo verificado es **129,990034 s**, del modelo de atención de
subtrayectos. Ver [actualización y dependencias](ACTUALIZACION_129_99.md).
Se incluyen su checkpoint y los dos padres requeridos. Los resultados, checkpoints
y pilotos nuevos no deben confundirse con los estados históricos de la primera entrega.

## Cifras que deben distinguirse""",1)
guide=guide.replace("| Mejor modelo individual conservado | 131,256712 | Misma Validation; checkpoint incluido |",
"""| Mejor modelo actual: atención de subtrayectos | **129,990034** | Misma Validation; cadena de tres checkpoints incluida |
| Modelo original conservado | 131,256712 | Misma Validation; checkpoint incluido |""")
guide=guide.replace("## Modelos incluidos","## Modelos de la primera entrega, también conservados",1)
guide=guide.replace("completa estaba en curso al capturar la entrega. No se le atribuye una mejora.",
"""completa estaba en curso en la primera captura. Sus resultados posteriores están en
   el registro actualizado de ese experimento; no es el nuevo mejor modelo.""")
text("revision_tesis/README.md",guide)
# Extend the existing independent metric checker to completed primary outputs of this stage.
checker=(OUT/"revision_tesis/verificar_resultados.py").read_text(encoding="utf-8")
checker=checker.replace('ROOT/"propuesta/comparaciones_transfertraj"]',
                        'ROOT/"propuesta/comparaciones_transfertraj",ROOT/"propuesta/replanteamiento_20261121"]')
checker=checker.replace('if "validation" not in record:continue',
                        'if "validation" not in record or record["validation"].get("n")!=127941:continue')
# Some historical files store other named outputs. They are not the primary per-run prediction set.
checker=checker.replace('pred=np.asarray(data["pred"],dtype=np.float64)',
'''if not {"pred","true"}.issubset(data.files):continue
                pred=np.asarray(data["pred"],dtype=np.float64)''')
text("revision_tesis/verificar_resultados.py",checker)
text("revision_tesis/verificar_nuevo_modelo.py",'''"""Read-only CPU inspection of the complete saved checkpoint chain; no training or Test access."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DEV=ROOT/"propuesta/replanteamiento_20261121"
CHAIN=[("historica_fresca_s0",27),("atencion_historia_individual_s0",3),("atencion_subtrayectos_s0",6)]
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
def main():
    import torch
    torch.set_num_threads(2)
    checks=[]
    for run_id,epoch in CHAIN:
        run=DEV/run_id
        record=json.loads((run/"resultado.json").read_text(encoding="utf-8-sig"))
        path=run/f"epoca_{epoch}.pt"
        assert digest(path)==record["model_sha256"]
        state=torch.load(path,map_location="cpu",weights_only=False)
        assert state["epoch"]==epoch,(run_id,state.keys())
        assert "state" in state or "branch_state" in state
        for name,expected in record.get("sources",{}).items():
            assert digest(run/"fuentes"/name)==expected,(run_id,name)
        checks.append({"model":run_id,"epoch":epoch,"sha256":record["model_sha256"]})
        del state
    run=DEV/CHAIN[-1][0]
    record=json.loads((run/"resultado.json").read_text(encoding="utf-8-sig"))
    proof=json.loads((run/"reproduccion_independiente.json").read_text(encoding="utf-8-sig"))
    assert proof["passed"] and proof["all127941quantiles_exact"] and proof["raw65_reverse_singleton_exact"]
    with np.load(run/"validacion.npz",allow_pickle=False) as data:
        assert len(data["pred"])==len(data["true"])==127941
        assert len(np.unique(data["row"]))==127941
        mae=float(np.abs(data["pred"].astype(float)-data["true"].astype(float)).mean())
    assert abs(mae-record["validation"]["mae_s"])<1e-9
    assert abs(mae-129.99003446901307)<1e-9
    print(json.dumps({"passed":True,"checkpoints":checks,"n":127941,"MAE_s":mae,
                      "saved_full_reproduction_passed":True,
                      "new_inference_executed":False,"GPU_used":False,"Test_read":False},indent=2))
if __name__=="__main__":main()
''')
text("revision_tesis/procedencia_actualizacion.json",json.dumps({
 "snapshot_utc":snapshot,"best_model":best["id"],"MAE_s":best["validation"]["mae_s"],
 "copied_files":copied,"source_project_modified":False,"training_or_GPU_used":False,
 "external_assets_not_uploaded":external
},ensure_ascii=False,indent=2))
# Scan newly copied text for common literal credentials; never upload execution environments.
patterns=[re.compile(r"gh[pousr]_[A-Za-z0-9]{25,}"),re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
          re.compile(r"hf_[A-Za-z0-9]{30,}"),re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{35,}"),
          re.compile(r"AKIA[0-9A-Z]{16}"),re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")]
for item in copied:
    path=OUT/item["path"]
    if path.suffix.lower() not in (".py",".json",".csv",".md"):continue
    value=path.read_text(encoding="utf-8-sig",errors="replace")
    assert not any(pattern.search(value) for pattern in patterns),item["path"]
manifest=[]
for path in sorted(OUT.rglob("*")):
    if not path.is_file() or ".git" in path.parts:continue
    rel=path.relative_to(OUT)
    if rel.as_posix()=="revision_tesis/manifiesto_archivos.json":continue
    if rel.parts[0] not in ("propuesta","auditoria","revision_tesis") and rel.name not in ("README.md","requirements-revision.txt",".gitignore",".gitattributes"):continue
    assert path.stat().st_size<95*1024**2,(rel,path.stat().st_size)
    manifest.append({"path":rel.as_posix(),"bytes":path.stat().st_size,"sha256":sha(path)})
text("revision_tesis/manifiesto_archivos.json",json.dumps({
 "files":manifest,"file_count":len(manifest),"total_bytes":sum(x["bytes"] for x in manifest),
 "raw_Train_Test_and_environment_excluded":True,"secret_literal_scan_passed":True,
 "original_repository_files_not_rehashed_as_research_artifacts":True
},ensure_ascii=False,indent=2))
print(json.dumps({"files":len(manifest),"MiB":sum(x["bytes"] for x in manifest)/1024**2,
 "new_selected_models":selected,"best_MAE_s":best["validation"]["mae_s"],
 "full_inference_resources_external":len(external)},indent=2))
