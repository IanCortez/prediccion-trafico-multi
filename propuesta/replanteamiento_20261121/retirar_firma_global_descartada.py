"""Remove one reconstructed-only teacher after the negative diffusion family closed."""
from base_replanteo import *
run=OUT/'difusion_global_tiempo_geometria_s0';r=read(run/'resultado.json')
assert r['complete'] and not r['qualified'] and not r['meaningful_gate_passed']
assert sha(run/'validacion.npz')==r['predictions_sha256']
folder=OUT/'datos_firma_global90_Train_s0';m=read(folder/'resultado.json');p=(folder/'firma90_Train.npz').resolve()
assert p.is_relative_to(OUT.resolve()) and p.exists() and sha(p)==m['outputs_sha256']
# Current source family uses rawGPS points; no signature file dependency.
for name in ['tokens_subtrayectos.py','entrenar_atencion_subtrayectos.py','normalizar_tokens_subtrayectos.py']:
 assert 'firma90_Train.npz' not in (OUT/name).read_text()
size=p.stat().st_size;digest=sha(p);p.unlink()
write(folder/'intermediario_retirado.json',dict(file=str(p),sha256=digest,bytes=size,MiB=size/1024**2,
 reason='Privategeometryteacher for completednegative diffusion; noqualified modeldepends onit.',
 recreate_from='preparar_firma_global90_Train.py + firma_global90.py and unchangedcoreteacher',
 original_generation_seconds=m['seconds'],all_completed_metrics_predictions_sources_retained=True))
print(json.dumps(dict(removed_MiB=size/1024**2,predictions_preserved=True)))
