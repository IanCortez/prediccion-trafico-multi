"""Read Train/road NPZ headers and existing mapper sources; no GPS/Test records decoded."""
from base_replanteo import *
import zipfile,ast
start=time.time();files={}
targets=[DATA/'train_gps_mapmatched.npz',DATA/'mapa_dirigido_2014.npz',DATA/'grafo_compacto_2014.npz',OUT/'datos_semi_markov/runs_train.npz']
for path in targets:
 assert path.resolve().is_relative_to(ROOT.resolve())
 if not path.exists():continue
 arrays={}
 with zipfile.ZipFile(path) as archive:
  for name in archive.namelist():
   if not name.endswith('.npy'):continue
   with archive.open(name) as f:
    version=np.lib.format.read_magic(f)
    shape,fortran,dtype=(np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f))
   arrays[name[:-4]]=dict(shape=shape,dtype=str(dtype),fortran=fortran,contains_object=dtype.hasobject)
 files[str(path.relative_to(ROOT))]=dict(bytes=path.stat().st_size,arrays=arrays,read='OnlyNPYheaders, no data arrays or pickle decoded.')
mappers=[]
for p in sorted(FROZEN.glob('*.py')):
 text=p.read_text(encoding='utf-8-sig')
 if not any(word in text.lower() for word in ['viterbi','map_match','mapmatch']):continue
 try:tree=ast.parse(text)
 except SyntaxError:continue
 names=[node.name for node in ast.walk(tree) if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and any(w in node.name.lower() for w in ['match','viterbi','profile','route'])]
 if names:mappers.append(dict(file=str(p.relative_to(ROOT)),sha256=sha(p),definitions=names))
raw=ROOT/'tesis/porto/porto/train.csv'
result=dict(scope='Prerequisite audit for Porto adaptation, not an experiment or source-label validation.',
 archives=files,mapper_sources=mappers,raw_original_csv=dict(exists=raw.is_file(),bytes=raw.stat().st_size if raw.exists() else None,opened=False),
 original_primary_labels='Unchanged(N−1)*15; every primary query remains. Route/segment labels Trainauxiliary only.',
 next_required=['Inspect actual graph/mapper schemas before deciding whether street sequences already exist or must be rebuilt.',
 'When rebuilding from shared rawCSV: binaryline index membership in Trainrows BEFORE decoding/parsingPOLYLINE. Never decode heldout/Test raw paths.',
 'Project input endpoints only from raw5/publicmap; actual mapmatched endpoints are auxiliary labels, never query inputs.',
 'UNITEOD forward needs explicit absorbing terminal semantics and a length-independent cap; verify batch/singleton invariance before training.',
 'No pretrained network or PortoMAE claim arises from syntheticcontrolflow proof.'],
 Test_access=False,real_GPS_arrays_decoded=False,source_sha256=sha(__file__),seconds=time.time()-start)
write(OUT/'insumos_unite_porto.json',result)
print(json.dumps(dict(archives_checked=len(files),mapper_sources=len(mappers),raw_csv_present=raw.is_file(),raw_GPS_or_Test_decoded=False,seconds=result['seconds'])),flush=True)
