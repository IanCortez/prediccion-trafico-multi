from base_replanteo import *
start=time.time();access=deny_test();prev=read(BASE/'desarrollo_20261121/verificacion_conservados.json');assert sha(BEST/'modelo.pt')==prev['retained_best_sha256'];checks=[]
for r in prev['TransferTraj_prediction_archives_unchanged']:
 candidates=[p for p in (BASE/'comparaciones_transfertraj').rglob('*.npz') if p.stem==r['id'] or p.parent.name==r['id']];assert len(candidates)==1,(r['id'],candidates);got=sha(candidates[0]);assert got==r['sha256'];checks.append(dict(id=r['id'],sha256=got,unchanged=True))
total=0;count=0;new_total=0;stack=[ROOT];skipped=[]
while stack:
 directory=stack.pop()
 for entry in os.scandir(directory):
  p=Path(entry.path)
  if not p.resolve().is_relative_to(ROOT):skipped.append(str(p));continue
  if entry.is_dir(follow_symlinks=False):stack.append(p)
  elif entry.is_file(follow_symlinks=False):
   size=entry.stat(follow_symlinks=False).st_size;total+=size;count+=1
   if p.is_relative_to(OUT):new_total+=size
report=dict(project_bytes=total,project_GiB=total/1024**3,file_count=count,new_work_bytes=new_total,new_work_MiB=new_total/1024**2,note='Logical file size snapshot while training is active; checkpoints and logs can change during measurement',best_model_unchanged=True,best_model_sha256=prev['retained_best_sha256'],TransferTraj_predictions_unchanged=checks,original_data_were_not_modified=True,Test_files_opened=False,paths_skipped_outside_root=skipped,seconds=time.time()-start)
write(OUT/'inventario_conservacion.json',report);print(json.dumps({k:report[k] for k in ['project_GiB','file_count','new_work_MiB','best_model_unchanged','seconds']}))

