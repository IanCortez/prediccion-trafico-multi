"""Prepare a reviewed retention/deletion manifest. This script NEVER deletes files."""
from pathlib import Path
import os,json,hashlib,time,shutil,zipfile,csv,collections
import numpy as np
ROOT=Path(r'E:\UTEC\Tesis\TesisIan').resolve();BASE=ROOT/'propuesta';DATA=BASE/'datos';RUNS=BASE/'experimentos';AUDIT=ROOT/'auditoria/limpieza';RESULTS=BASE/'resultados';CMP=BASE/'comparaciones_transfertraj';BEST='od_mezcla8_geometria_conjunta_s0_fp32_v2'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def files(folder):
    for directory,dirs,names in os.walk(folder,followlinks=False):
        for name in list(dirs):
            p=Path(directory)/name
            assert not (p.stat(follow_symlinks=False).st_file_attributes&0x400),str(p)
        for name in names:
            p=Path(directory)/name;assert not (p.stat(follow_symlinks=False).st_file_attributes&0x400),str(p)
            yield p


def size(path):
    if path.is_file():return path.stat().st_size,1
    n=total=0
    for p in files(path):total+=p.stat().st_size;n+=1
    return total,n


def main():
    assert Path(os.sys.executable).resolve().is_relative_to(ROOT/'.conda-env');proof=json.loads((AUDIT/'verificacion_antes.json').read_text());assert proof['passed'] and proof['all_validation_predictions_bit_identical'];assert not (AUDIT/'plan_eliminacion.json').exists();RESULTS.mkdir(exist_ok=True);CMP.mkdir(exist_ok=True)
    before_size,before_files=size(ROOT);write(AUDIT/'espacio_antes.json',dict(bytes=before_size,GiB=before_size/2**30,files=before_files))
    best=RUNS/BEST;record=json.loads((best/'resultado.json').read_text());assert sha(best/'modelo.pt')==record['model_sha256'];gold=np.load(best/'validacion.npz');baseline=[];all_results=[]
    for folder in sorted(RUNS.iterdir()):
        if not folder.is_dir():continue
        rec={};record_name=None
        for name in ['resultado.json','resultado_preentrenamiento.json','ejecucion.json']:
            p=folder/name
            if p.exists():rec=json.loads(p.read_text(encoding='utf-8-sig'));record_name=name;break
        config=rec.get('config',{});architecture=rec.get('architecture',config.get('architecture',''));is_transfer=('transfertraj' in folder.name.lower() or 'transfertraj' in architecture.lower() or config.get('transfertraj_resources',False));validation=rec.get('validation')
        if validation:all_results.append(dict(id=folder.name,architecture=architecture,cohort=rec.get('cohort','primary'),validation=validation,config=config,model_retained=folder.name==BEST,source_record=str((folder/record_name).relative_to(ROOT))))
        if not is_transfer:continue
        target=CMP/folder.name;target.mkdir();saved=[]
        for p in folder.iterdir():
            if p.is_file() and p.suffix=='.json':shutil.copy2(p,target/p.name);saved.append(p.name)
        prediction=folder/'validacion.npz';verified=None
        if prediction.exists():
            z=np.load(prediction);assert 'pred' in z and 'true' in z and 'row' in z;assert np.array_equal(z['row'],gold['row']) and np.array_equal(z['true'],gold['true']),folder.name
            if 'timestamp' in z:assert np.array_equal(z['timestamp'],gold['timestamp'])
            mae=float(np.abs(z['true'].astype(float)-z['pred'].astype(float)).mean());metrics=validation
            if metrics is None and (folder/'mejor.json').exists():metrics=json.loads((folder/'mejor.json').read_text()).get('validation')
            assert metrics and abs(mae-metrics['mae_s'])<1e-9,(folder.name,mae,metrics)
            shutil.copy2(prediction,target/prediction.name);assert sha(prediction)==sha(target/prediction.name);saved.append(prediction.name);verified=dict(n=len(z['row']),MAE_s=mae,source_SHA256=sha(prediction),predictions_and_rows_and_truth_retained_without_changes=True,keys=list(z.files))
        baseline.append(dict(id=folder.name,architecture=architecture,stage=config.get('stage'),status='completed' if (folder/'resultado.json').exists() else 'pretraining_completed' if (folder/'resultado_preentrenamiento.json').exists() else 'incomplete',validation=validation,predictions=verified,retained_files=saved,weights_retained=False,not_exact_published_Porto_cohort=True))
    assert sum(v['predictions'] is not None for v in baseline)==10
    write(CMP/'indice.json',dict(protocol='Same_local_primary_validation_127941_trips;no_Test_predictions',published_TransferTraj_323481_trip_cohort_not_available=True,not_a_replication_of_published_73_5_seconds=True,cases=baseline))
    write(RESULTS/'resultados_primarios.json',all_results)
    with (RESULTS/'resultados_primarios.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['id','architecture','cohort','validation_n','mae_s','rmse_s','model_retained'])
        for r in all_results:writer.writerow([r['id'],r['architecture'],r['cohort'],r['validation']['n'],r['validation']['mae_s'],r['validation']['rmse_s'],r['model_retained']])
    internal=[]
    for p in (BASE/'instrumentos/diagnosticos').rglob('resultado.json'):
        if 'fuentes' in p.parts:continue
        z=json.loads(p.read_text(encoding='utf-8-sig'))
        if 'metrics' in z:internal.append(dict(source=str(p.relative_to(ROOT)),metrics=z['metrics'],comparisons=z.get('comparisons'),eligible_primary_followup=z.get('eligible_primary_followup'),scope='Internal_development_NOT_primary_validation_or_Test'))
    write(RESULTS/'resultados_internos.json',internal)
    original=RUNS/'original_completo_pace_pos_s0';target=RESULTS/'original_corregido';target.mkdir()
    for name in ['resultado.json','evaluacion_critica.json','validacion.npz','mejor.json']:
        if (original/name).exists():shutil.copy2(original/name,target/name)
    provenance=best/'procedencia_entrenamiento';provenance.mkdir()
    for name in ['resultado.json','ejecucion.json','historia.json','mejor.json']:
        p=RUNS/'od_mezcla8_geometria_conjunta_s0'/name
        if p.exists():shutil.copy2(p,provenance/name)
    # Deduplicated historical evidence: code and small metadata, never tensor caches/models.
    archive=RESULTS/'historial_evidencia.zip';objects=set();index={};archived_bytes=0
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as out:
        for scope in [BASE,ROOT/'auditoria']:
            for directory,dirs,names in os.walk(scope,followlinks=False):
                here=Path(directory)
                dirs[:]=[n for n in dirs if n not in ['__pycache__','.git'] and not ((here/n)==RESULTS or (here/n)==CMP or (here/n)==AUDIT or (here/n)==best or (here/n)==ROOT/'auditoria/muestras' or (here/n)==ROOT/'auditoria/revision_bibliografica' or (here/n)==BASE/'referencias' or (here/n)==BASE/'modelos')]
                for name in names:
                    p=here/name
                    if p.suffix.lower() not in ['.py','.json','.jsonl','.md','.svg','.ps1','.yaml','.yml']:continue
                    if p.stat().st_size>10*2**20:continue
                    if p.is_relative_to(DATA) and p.suffix.lower()!='.json':continue
                    data=p.read_bytes();h=hashlib.sha256(data).hexdigest();object_name='objects/'+h
                    if h not in objects:out.writestr(object_name,data);objects.add(h);archived_bytes+=len(data)
                    index[str(p.relative_to(ROOT))]=dict(SHA256=h,object=object_name,bytes=len(data))
        out.writestr('index.json',json.dumps(index,ensure_ascii=False))
        out.writestr('LEEME.txt','Archivo historico desduplicado de fuentes y evidencia ligera. index.json vincula cada ruta original con un objeto SHA256. Los datos derivados masivos y modelos descartados no estan en este archivo. Estados antiguos son historicos: todas las ejecuciones se detuvieron por peticion del usuario.')
    with zipfile.ZipFile(archive) as check:assert check.testzip() is None
    write(RESULTS/'historial_evidencia.json',dict(SHA256=sha(archive),original_paths=len(index),unique_objects=len(objects),unique_uncompressed_bytes=archived_bytes,archive_bytes=archive.stat().st_size,no_discarded_tensor_caches_or_model_weights=True))
    # Preserve exactly the best model's cached training/evaluation inputs and raw-query resources.
    keep_data={'manifiesto.json','geometria_train.npz','geometria_validation.npz','mapa_dirigido.npz','mapa_dirigido_2014.npz','grafo_compacto.npz','grafo_compacto_2014.npz','contexto_rejilla25m.npz','distancias_2014','distancias_2014_free','perfiles_mapmatched_2014','mapa_manifiesto.json','mapa_2014_manifiesto.json','grafo_compacto_manifiesto.json','contexto_manifiesto.json','geometria_manifiesto.json','mapmatched_features_manifiesto.json','ruteo_compacto_manifiesto.json'}
    for split in ['train','validation','test']:
        keep_data.update({split+'_x.npz',split+'_y.npy',split+'_rutas.npz',split+'_contexto.npz',split+'_rutas_proyectadas2014.npz',split+'_gps_mapmatched.npz'})
    for name in proof['actual_data_files_accessed']:assert Path(name).parts[2] in keep_data,name
    original_files=list(files(ROOT/'tesis'))+list(files(ROOT/'codigo-ian'))+list(files(ROOT/'auditoria/muestras'))
    protected={str(p.relative_to(ROOT)):sha(p) for p in original_files}
    for p in files(best):protected[str(p.relative_to(ROOT))]=sha(p)
    for name in keep_data:
        p=DATA/name
        if not p.exists():continue
        for q in files(p) if p.is_dir() else [p]:protected[str(q.relative_to(ROOT))]=sha(q)
    for folder in [CMP,BASE/'referencias/mapas',BASE/'referencias/TransferTraj']:
        for p in files(folder):protected[str(p.relative_to(ROOT))]=sha(p)
    protected[str(archive.relative_to(ROOT))]=sha(archive)
    write(AUDIT/'archivos_protegidos.json',dict(files=protected,original_files_n=len(original_files),raw_originals_and_best_and_TransferTraj_predictions_preserved=True,Test_labels_not_decoded=True))
    candidates=[]
    def prune(folder,keep):
        for p in folder.iterdir():
            if p.name not in keep:candidates.append(p)
    prune(BASE,{'datos','experimentos','referencias','instrumentos','comparaciones_transfertraj','resultados','reproducir_mejor.py','README.md','PROTOCOLO.md','RESUMEN_RESULTADOS.md','ARQUITECTURA.md','REPRODUCIR.md'})
    prune(DATA,keep_data);prune(RUNS,{BEST});prune(BASE/'referencias',{'mapas','TransferTraj','TransferTraj_suplemento.zip','suplemento_archivos.json'});prune(BASE/'instrumentos',set());prune(ROOT/'auditoria',{'limpieza','muestras','revision_bibliografica','entorno.ps1','conda-explicito.txt','dependencias-exactas.txt'});prune(ROOT/'.runtime',{'condarc.yml'})
    candidates.extend((ROOT/'.conda-env').rglob('__pycache__'))
    selected=[]
    for p in sorted(set(candidates),key=lambda p:len(p.parts)):
        resolved=p.resolve();assert resolved.is_relative_to(ROOT) and resolved!=ROOT
        if any(resolved.is_relative_to(q) for q in selected):continue
        assert not any((ROOT/name).is_relative_to(resolved) for name in protected),(p,'protected file below target')
        selected.append(resolved)
    deletion=[]
    for p in selected:
        count_bytes,count_files=size(p);deletion.append(dict(path=str(p),relative_path=str(p.relative_to(ROOT)),is_directory=p.is_dir(),bytes=count_bytes,files=count_files))
    assert sum(r['bytes'] for r in deletion)>50*2**30
    plan=dict(workspace=str(ROOT),user_request='Remove caches, duplicates and discarded intermediates;retain original data,best-model reproduction,summary,TransferTraj metrics/predictions',protected_manifest_SHA256=sha(AUDIT/'archivos_protegidos.json'),archive_SHA256=sha(archive),best_precleanup_verification_SHA256=sha(AUDIT/'verificacion_antes.json'),targets=deletion,planned_delete_bytes=sum(r['bytes'] for r in deletion),planned_delete_files=sum(r['files'] for r in deletion),TransferTraj_prediction_sets=10,no_deletion_performed_by_this_script=True)
    write(AUDIT/'plan_eliminacion.json',plan)
    print(json.dumps(dict(preparation_passed=True,before_GiB=before_size/2**30,planned_delete_GiB=plan['planned_delete_bytes']/2**30,delete_targets=len(deletion),protected_files=len(protected),TransferTraj_prediction_sets=10,archived_metadata_paths=len(index),unique_archive_objects=len(objects),archive_MiB=archive.stat().st_size/2**20)),flush=True)

if __name__=='__main__':main()
