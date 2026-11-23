from base_replanteo import *
p=OUT/'inferencia_atencion_subtrayectos.py';s=p.read_text()
s=s.replace("module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module",
 "module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module")
s=s.replace("source=frozen_module('recuperar_subtrayectos')",
 "source=importlib.import_module('recuperar_subtrayectos')\n  assert Path(source.__file__).resolve()==(run/'fuentes/recuperar_subtrayectos.py').resolve()")
p.write_text(s)
folder=(ROOT/'.runtime/numba/fuentes_99078463e5334c2c356bc1e1ecdfaed6bbab64ee').resolve()
assert folder.is_relative_to(ROOT/'.runtime/numba') and folder.is_dir()
removed=[]
for file in folder.glob('recuperar_subtrayectos.*'):
 assert file.resolve().parent==folder and file.suffix in ['.nbc','.nbi']
 removed.append(dict(file=str(file),sha256=sha(file),bytes=file.stat().st_size));file.unlink()
write(OUT/'recuperacion_cache_subtrayectos.json',dict(cause='Numba cached a function from an unregistered dynamic module; secondprocess could not rebuild its moduleenvironment.',
 fix='Use regular frozen-source module import and register explicit auxiliarymodules in sys.modules. Recompile only4localfunctioncachefiles.',
 removed_cache_only=removed,no_model_architecture_or_weights_changed=True,full_evaluation_and_training_not_repeated=True))
print(json.dumps(dict(cachefiles_removed=len(removed),source_import_corrected=True)))
