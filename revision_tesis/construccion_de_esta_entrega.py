from pathlib import Path
import hashlib,json,shutil,sys,time,zipfile,types,re
import numpy as np

ROOT=Path(r"E:\UTEC\Tesis\TesisIan")
OUT=ROOT/"publicacion_github_20261122"
BEST="od_mezcla8_geometria_conjunta_s0_fp32_v2"
assert (OUT/".git").is_dir()
def sha_bytes(data): return hashlib.sha256(data).hexdigest()
copied=[]
def copy(source,relative=None):
    source=Path(source)
    relative=Path(relative) if relative else source.relative_to(ROOT)
    for attempt in range(3):
        payload=source.read_bytes()
        if sha_bytes(source.read_bytes())==sha_bytes(payload): break
    else: raise RuntimeError(f"Source changing during snapshot: {source}")
    target=OUT/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(payload)
    copied.append({"path":relative.as_posix(),"source_sha256":sha_bytes(payload),"bytes":len(payload)})
def text(relative,payload):
    target=OUT/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(payload,encoding="utf-8")
for name in ("ARQUITECTURA.md","PROTOCOLO.md","RESUMEN_RESULTADOS.md","README.md","REPRODUCIR.md","reproducir_mejor.py","reentrenar_mejor.py"):
    copy(ROOT/"propuesta"/name)
for source in (ROOT/"propuesta/experimentos"/BEST).rglob("*"):
    if source.is_file() and "__pycache__" not in source.parts and source.suffix not in (".pyc",):
        copy(source)
for folder in ("comparaciones_transfertraj","desarrollo_20261121","replanteamiento_20261121"):
    for source in sorted((ROOT/"propuesta"/folder).rglob("*")):
        if not source.is_file(): continue
        relative_parts=source.relative_to(ROOT/"propuesta"/folder).parts[:-1]
        if any(part.startswith(("datos","cache","__pycache__")) or part in (".git","tmp") for part in relative_parts): continue
        if source.suffix.lower() in (".py",".md",".json",".csv",".svg",".png"):
            if source.stat().st_size<10*1024**2: copy(source)
        elif source.suffix==".npz" and (source.name.startswith(("validacion","predicciones")) and "progreso" not in source.name):
            if source.stat().st_size<45*1024**2: copy(source)
for folder in ("resultados",):
    for source in (ROOT/"propuesta"/folder).rglob("*"):
        if source.is_file() and source.suffix.lower() in (".md",".json",".csv",".zip",".npz"):
            if source.stat().st_size<45*1024**2: copy(source)
for source in (ROOT/"propuesta/referencias/TransferTraj").rglob("*"):
    if source.is_file() and not any(x in source.parts for x in (".git","__pycache__","data","dataset","datasets")):
        if source.suffix.lower() in (".py",".md",".txt",".yaml",".yml",".toml",".json") or source.name.startswith("LICENSE"):
            if source.stat().st_size<1024**2: copy(source)
for name in ("dependencias-exactas.txt","conda-explicito.txt"):
    copy(ROOT/"auditoria"/name)
for name in ("preparar_limpieza.py","resumen_limpieza.json","eliminados.jsonl"):
    copy(ROOT/"auditoria/limpieza"/name)
copy(ROOT/"auditoria/revision_bibliografica/ANALISIS_PROPUESTA.md")
for source in (ROOT/"propuesta/datos").glob("*manifiesto*.json"):
    copy(source)
# Only stable, already saved weights; never copy optimizer/resume states.
extra_models={
    "espacial_fases_referencia_s0":"Complete Validation, MAE 132.700776 s; does not improve baseline.",
    "espacial_prior_cf_s0":"Complete Validation, MAE 137.538866 s; does not improve baseline.",
    "prefijo_nativo_s0":"Training complete; full Validation evaluation pending at snapshot. Not a qualified candidate."
}
for run,status in extra_models.items():
    source=ROOT/"propuesta/replanteamiento_20261121"/run/"modelo.pt"
    if source.exists(): copy(source)
# Recover the ensemble record from lightweight historical evidence, without extracting the archive.
with zipfile.ZipFile(ROOT/"propuesta/resultados/historial_evidencia.zip") as archive:
    index=json.loads(archive.read("index.json"))
    record=json.loads(archive.read(index[r"propuesta\ESTADO.json"]["object"]))
    ensemble=record["best_distribution_ensemble"]
text("revision_tesis/ensamble_historico_131_06.json",json.dumps({
    "record":ensemble,
    "status":"Historical exploratory Validation result. Three of four component checkpoints were removed on 2026-11-19; this ensemble cannot be loaded from this delivery.",
    "not_a_new_model_or_Test_result":True
},ensure_ascii=False,indent=2))
# Prepare a small example using identical frozen feature code, never importing common.py or running inference.
frozen=ROOT/"propuesta/experimentos"/BEST/"fuentes"
stub=types.ModuleType("common")
stub.np=np
stub.Path=Path
stub.ROOT=ROOT
stub.BASE=ROOT/"propuesta"
stub.DATA=stub.BASE/"datos"
stub.json=json
stub.os=__import__("os")
stub.sys=sys
stub.hashlib=hashlib
stub.time=time
sys.modules["common"]=stub
sys.path.insert(0,str(frozen))
from features import features
cfg=json.loads((ROOT/"propuesta/experimentos"/BEST/"resultado.json").read_text(encoding="utf-8-sig"))
with np.load(ROOT/"propuesta/datos/validation_x.npz",allow_pickle=False) as data:
    d={key:data[key][:65] for key in data.files}
X,names=features(d,calendar=True)
for component in ("rutas","contexto","rutas_proyectadas2014","gps_mapmatched"):
    with np.load(ROOT/f"propuesta/datos/validation_{component}.npz",allow_pickle=False) as data:
        assert np.array_equal(d["row"],data["row"][:65])
        X=np.column_stack((X,data["X"][:65]))
        names+=data["names"].tolist()
assert names==cfg["features"] and X.shape==(65,216)
with np.load(ROOT/"propuesta/experimentos"/BEST/"validacion.npz",allow_pickle=False) as data:
    example=dict(X=X,names=np.asarray(names),row=data["row"][:65],expected_pred=data["pred"][:65],true=data["true"][:65])
np.savez_compressed(OUT/"revision_tesis/ejemplo_65_validacion.npz",**example)
text("requirements-revision.txt","numpy==1.26.4\ntorch==2.5.1\ntabm==0.0.3\nrtdl_num_embeddings==0.0.12\nscikit-learn==1.5.2\nscipy==1.14.1\n")
text("propuesta/datos/README.md","""# Recursos externos necesarios
Los conjuntos completos de Train/Validation/Test, GPS crudos, mapas binarios y caches no se publican en Git.
Los manifiestos conservan su procedencia y las huellas. La muestra de 65 entradas derivadas de Validation
está en ../../revision_tesis/ejemplo_65_validacion.npz y permite inspeccionar el checkpoint.
Las métricas completas se pueden recalcular desde los archivos de predicciones conservados.
Para reproducir desde consultas OD o reentrenar, se necesitan además los recursos originales de la
carpeta local del proyecto, manteniendo rutas y SHA256 documentados en cada ejecución.
No se publicaron ni evaluaron etiquetas de Test como parte de esta entrega.
""")
text("README.md","""# Predicción de tráfico: avances para revisión de tesis

Esta rama conserva el código original del alumno y añade una instantánea de investigación para su revisión.
La guía de entrada es [revision_tesis/README.md](revision_tesis/README.md).

- [Resultados, modelos y límites de interpretación](revision_tesis/README.md).
- [Resumen de experimentos](propuesta/RESUMEN_RESULTADOS.md).
- [Comparaciones locales con TransferTraj](propuesta/comparaciones_transfertraj/README.md).
- [Arquitectura del modelo conservado](propuesta/ARQUITECTURA.md).
- [Nuevo replanteamiento y experimentos negativos](propuesta/replanteamiento_20261121/README.md).
- [Evidencia del ensamble histórico de 131,06 s](revision_tesis/ensamble_historico_131_06.json).

**Meta MAE <73,5 s: pendiente.** Mejor modelo individual conservado: **131,256712 s** en
127.941 viajes de Validation. El ensamble histórico llegó a **131,063381 s**, pero tres
de sus cuatro checkpoints se eliminaron en la limpieza previa. Se conserva su registro; no se afirma
que ese ensamble pueda cargarse desde esta rama. Test no fue evaluado.

Esta entrega es una instantánea: los estados de procesos y las cifras parciales son históricos.
""")
text("revision_tesis/README.md","""# Guía para Ian: revisión de resultados y modelos

Instantánea de investigación del 22 de noviembre de 2026. Los archivos originales src/ se mantienen.
La rama aporta código, pesos retenidos, métricas, predicciones, protocolos y evidencia de resultados
negativos. Las búsquedas usaron Validation de forma adaptativa: las cifras no son confirmación de Test.

## Cifras que deben distinguirse

| Sistema | MAE en segundos | Alcance |
|---|---:|---|
| Código original corregido | 140,822594 | 127.941 viajes de Validation |
| Mejor modelo individual conservado | 131,256712 | Misma Validation; checkpoint incluido |
| Ensamble histórico | 131,063381 | Misma Validation; registro incluido, componentes incompletos |
| TransferTraj adaptado sin preentrenamiento | 164,367686 | Misma Validation |
| TransferTraj adaptado con preentrenamiento de 30 épocas | 171,720112 | Misma Validation |
| Variante de TransferTraj: reloj preciso y duración MAE | 158,877006 | Misma Validation |
| Nuevo modelo con rutas globales | 131,399512 | Misma Validation; no mejoró |
| Estado causal con información de viajes ya completados | 130,418913 | Protocolo online diferente |

Los 73,5 s del artículo TransferTraj corresponden a otra cohorte de Porto (323.481 viajes).
No se obtuvo esa cohorte; aquí se conservaron 1.279.412 viajes con filtros explícitos.
Estos controles locales adaptan recursos y algunas decisiones del código oficial. No son una
reproducción exacta del resultado publicado ni demuestran transferencia entre ciudades.

Un resultado con selección privilegiada u oracle utiliza la respuesta real para elegir predicciones.
No es un modelo desplegable ni un MAE conseguido. Los pilotos internos de 4.096 o 512 casos tampoco
sustituyen la evaluación de los 127.941 viajes.

## Modelos incluidos

1. propuesta/experimentos/od_mezcla8_geometria_conjunta_s0_fp32_v2/modelo.pt:
   referencia completa, 4.280.768 parámetros, 216 atributos OD, ocho componentes de duración y
   geometría auxiliar solo durante entrenamiento. Incluye las 115 fuentes congeladas originales,
   configuración, procedencia de selección, historial y predicciones.
2. propuesta/replanteamiento_20261121/espacial_fases_referencia_s0/modelo.pt:
   alternativa con permanencias por fases; 132,700776 s. No superó la referencia.
3. propuesta/replanteamiento_20261121/espacial_prior_cf_s0/modelo.pt:
   alternativa direccional con prior de conteos; 137,538866 s. No superó la referencia.
4. propuesta/replanteamiento_20261121/prefijo_nativo_s0/modelo.pt:
   generador recurrente experimental. Entrenamiento de diez épocas terminado; la evaluación
   completa estaba en curso al capturar la entrega. No se le atribuye una mejora.

Cada peso se publica como binario de Git, por debajo de 100 MiB, con SHA256 en
manifiesto_archivos.json. Los pesos descartados de TransferTraj no se incluyen porque ya no existen;
sí se incluyen sus métricas y diez archivos de predicciones originales, uno de ellos incompleto y
marcado como tal. No confundir el intento incompleto con un modelo terminado.

## Cómo revisar sin los datos completos

Desde la raíz de esta rama, con NumPy disponible:

    python -B revision_tesis/verificar_resultados.py

Recalcula el MAE del checkpoint conservado y de los controles TransferTraj terminados desde
predicciones y etiquetas de Validation. Comprueba cantidades, alineación y huellas del paquete.
No entrena, no necesita GPU, no abre Test y no descarga datos.

Para inspeccionar el peso y hacer una inferencia pequeña sobre atributos ya preparados:

    python -B revision_tesis/probar_modelo.py --device cpu

La muestra de 65 consultas conserva el orden exacto de los 216 atributos. No contiene GPS interior.
Su MAE no sustituye la evaluación completa. CPU o distinto tamaño de lote pueden diferir de la
evaluación canónica GPU; la prueba informa esas diferencias y no redefine el resultado publicado.

## Entorno y reproducción

Se recomienda un entorno conda por prefijo dentro del checkout. En PowerShell:

    $env:CONDA_PKGS_DIRS = Join-Path (Get-Location) '.runtime/conda-pkgs'
    conda create --prefix ./.conda-env python=3.11 -y
    conda run --prefix ./.conda-env python -m pip install --cache-dir ./.runtime/pip-cache -r requirements-revision.txt
    conda run --prefix ./.conda-env python -B revision_tesis/verificar_resultados.py

requirements-revision.txt cubre los dos instrumentos de revisión. El entorno histórico completo
se conserva en auditoria/dependencias-exactas.txt y conda-explicito.txt como evidencia; contiene
paquetes y rutas de la máquina original, no es un instalador portable directo.

La reproducción canónica completa y el reentrenamiento requieren los datos preparados locales,
mapas, perfiles y dependencias originales: ver propuesta/REPRODUCIR.md. Los lanzadores conservados
esperan un entorno .conda-env bajo la raíz del checkout. Los scripts históricos con rutas absolutas
E:/UTEC/Tesis/TesisIan deben adaptarse en una copia de trabajo, manteniendo los archivos congelados.
Ningún entrenamiento se inicia al clonar o ejecutar los instrumentos de revisión.

## Lectura crítica para la tesis

- La mezcla de distribuciones mejoró modestamente al predictor directo con las mismas entradas;
  no basta para justificar novedad universal o superioridad sobre TransferTraj.
- Geometría más precisa, supervisión de bajo desplazamiento y memoria de vecinos no dieron
  mejoras consistentes en el error total. Los resultados negativos y sus controles se conservan.
- Menor pérdida local de trayectorias no demuestra menor MAE de viaje generado.
- Solo hay trayectorias completas de Porto para esta evaluación; faltan pruebas entre ciudades,
  repetición entre semillas y confirmación independiente de un candidato congelado.
- El ensamble de 131,06 s fue omitido de la política de conservación anterior. Sus tres componentes
  ausentes no están en el ZIP histórico: allí se archivaron fuentes y metadatos, no pesos.
- Test permanece reservado y el objetivo <73,5 s sigue sin alcanzarse.

El archivo propuesta/resultados/historial_evidencia.zip está desduplicado por contenido.
index.json asocia cada ruta original con un objeto SHA256 dentro del ZIP; no es un modelo cargable.
Las referencias de código externo conservan sus propios avisos/licencias y su atribución.
Los registros de procesos y rutas de máquina son evidencia de la captura, no instrucciones de ejecución.
""")
text("revision_tesis/verificar_resultados.py",'''"""Read-only review of saved Validation results; never trains, imports Torch or opens Test."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
def main():
    manifest=json.loads((ROOT/"revision_tesis/manifiesto_archivos.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path=ROOT/item["path"]
        assert path.is_file(),item["path"]
        assert path.stat().st_size==item["bytes"],item["path"]
        assert sha(path)==item["sha256"],item["path"]
    roots=[ROOT/"propuesta/experimentos",ROOT/"propuesta/comparaciones_transfertraj"]
    count=0
    for folder in roots:
        for record_path in sorted(folder.glob("*/resultado.json")):
            record=json.loads(record_path.read_text(encoding="utf-8-sig"))
            if "validation" not in record:continue
            if record.get("conclusion","").lower().find("incomplet")>=0:continue
            predictions=record_path.parent/"validacion.npz"
            if not predictions.exists():continue
            # The initial failed run is retained only as evidence, never as a completed model.
            if record_path.parent.name=="od_transfertraj_adaptado_sin_pre_s0":continue
            with np.load(predictions,allow_pickle=False) as data:
                pred=np.asarray(data["pred"],dtype=np.float64)
                true=np.asarray(data["true"],dtype=np.float64)
                assert pred.ndim==true.ndim==1 and len(pred)==len(true)==127941
                assert np.isfinite(pred).all() and np.isfinite(true).all()
                if "row" in data:assert len(np.unique(data["row"]))==len(pred)
                mae=float(np.abs(pred-true).mean())
            expected=float(record["validation"]["mae_s"])
            assert abs(mae-expected)<1e-5,(record_path.parent.name,mae,expected)
            print(f"{record_path.parent.name}: n={len(pred)} MAE={mae:.6f} s")
            count+=1
    assert count>=8,count
    print(f"OK: {len(manifest['files'])} file hashes; {count} completed Validation prediction sets. Test was not read.")
if __name__=="__main__":main()
''')
text("revision_tesis/probar_modelo.py",'''"""Inspect the retained model on a small prepared sample, CPU by default; no raw-query reconstruction."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BEST="od_mezcla8_geometria_conjunta_s0_fp32_v2"
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--n",type=int,default=8)
    args=parser.parse_args()
    assert 1<=args.n<=65
    import torch
    run=ROOT/"propuesta/experimentos"/BEST
    record=json.loads((run/"resultado.json").read_text(encoding="utf-8-sig"))
    path=run/"modelo.pt"
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record["model_sha256"]
    sys.path.insert(0,str(run/"fuentes"))
    from modelo_mezcla_rutas import make_duration_model
    # Only load the published trusted checkpoint after checking its recorded SHA256.
    ck=torch.load(path,map_location="cpu",weights_only=False)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=make_duration_model(ck,record["config"]).to(args.device)
    model.load_state_dict(ck["state"])
    model.eval()
    with np.load(ROOT/"revision_tesis/ejemplo_65_validacion.npz",allow_pickle=False) as data:
        assert data["names"].tolist()==record["features"]
        X=torch.as_tensor(data["X"][:args.n],device=args.device)
        expected=data["expected_pred"][:args.n]
    with torch.no_grad():
        output=model(X)
        probabilities=output[0].exp().mean(1)
        prediction=model.quantiles(output[0],(.5,))[:,0].cpu().numpy()
    assert torch.isfinite(probabilities).all()
    assert torch.allclose(probabilities.sum(-1),torch.ones(len(X),device=args.device),atol=1e-5)
    assert np.isfinite(prediction).all()
    print(json.dumps(dict(model_id=BEST,parameters=sum(p.numel() for p in model.parameters()),
                         device=args.device,n=args.n,prediction_s=prediction.tolist(),
                         historical_prediction_s=expected.tolist(),
                         max_difference_s=float(np.max(np.abs(prediction-expected))),
                         full_Validation_not_evaluated=True,Test_not_read=True),indent=2))
if __name__=="__main__":main()
''')
ignore=(OUT/".gitignore").read_text(encoding="utf-8")
text(".gitignore",ignore+"""
# Local research environments and raw/derived datasets
.conda-env/
.runtime/
publicacion_github_*/
**/*.pyc
**/ultima_reanudacion.pt
**/continuacion.pt
propuesta/datos/**/*.npy
propuesta/datos/**/*.npz
propuesta/datos/**/*.pbf
# Explicitly retain the compact evidence archive in this review snapshot
!propuesta/resultados/historial_evidencia.zip
""")
# Record all copied file provenance before creating the self-contained integrity manifest.
text("revision_tesis/procedencia_copia.json",json.dumps({
    "snapshot_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
    "repository":"IanCortez/prediccion-trafico-multi",
    "branch":"investigacion/resultados-modelos-2026-11-22",
    "original_main_commit":"b7052e01f374105e12929b9b5f5b0d9a3eaf6b80",
    "copy_only_no_original_source_mutations":True,
    "extra_checkpoint_status":extra_models,
    "files":copied
},ensure_ascii=False,indent=2))
# Secret-like literal scan, also scanning the deduplicated archive objects.
patterns=[re.compile(r"gh[pousr]_[A-Za-z0-9]{25,}"),re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
          re.compile(r"AKIA[0-9A-Z]{16}"),re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")]
secret_files=[]
for source in OUT.rglob("*"):
    if not source.is_file() or ".git" in source.parts:continue
    texts=[]
    if source.suffix.lower()==".zip":
        with zipfile.ZipFile(source) as z:
            texts=[z.read(n).decode("utf-8",errors="ignore") for n in z.namelist()]
    elif source.suffix.lower() in (".py",".json",".jsonl",".csv",".md",".txt",".ps1",".yaml",".yml"):
        texts=[source.read_text(encoding="utf-8-sig",errors="replace")]
    if any(pattern.search(value) for value in texts for pattern in patterns):
        secret_files.append(str(source.relative_to(OUT)))
assert not secret_files,secret_files
manifest=[]
for source in sorted(OUT.rglob("*")):
    if not source.is_file() or ".git" in source.parts:continue
    relative=source.relative_to(OUT).as_posix()
    if relative=="revision_tesis/manifiesto_archivos.json":continue
    size=source.stat().st_size
    assert size<95*1024**2,(relative,size)
    assert not any(part.startswith(".conda") or part==".runtime" for part in source.relative_to(OUT).parts)
    manifest.append({"path":relative,"bytes":size,"sha256":sha_bytes(source.read_bytes())})
text("revision_tesis/manifiesto_archivos.json",json.dumps({
    "files":manifest,
    "file_count":len(manifest),
    "total_bytes":sum(x["bytes"] for x in manifest),
    "raw_Train_Test_and_environment_excluded":True,
    "secret_literal_scan_passed":True
},ensure_ascii=False,indent=2))
print(json.dumps({"files":len(manifest),"MiB":sum(x["bytes"] for x in manifest)/1024**2,
                  "checkpoint_files":[x for x in manifest if x["path"].endswith("modelo.pt")],
                  "secret_scan_passed":True},indent=2))
