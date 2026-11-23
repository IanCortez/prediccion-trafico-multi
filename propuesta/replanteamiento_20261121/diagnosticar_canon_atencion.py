"""Inspect numerical source-order sensitivity before any neural training/Validation evaluation."""
from pathlib import Path
import ast,shutil
root=Path('E:/UTEC/Tesis/TesisIan/propuesta/replanteamiento_20261121')
archive=root/'verificacion_atencion_pre_canon';assert not archive.exists();archive.mkdir()
for name in ['modelo_atencion_historia.py','verificar_atencion_historia_cpu.py']:
 shutil.copy2(root/name,archive/name)
shutil.copy2(root/'verificar_atencion_historia_cpu.log',archive/'prueba_fallida.log')
s=(root/'verificar_atencion_historia_cpu.py').read_text(encoding='utf-8-sig')
needle=' assert error<2e-5';assert s.count(needle)==1
prefix=s[:s.index(needle)]
prefix+=""" result=dict(max_log_probability_difference=error,max_absolute_probability_difference=float((out[0].exp()-reverse[0].exp()).abs().max()),point_medians_exact=bool(torch.equal(model.duration.quantiles(out[0])[:,1],model.duration.quantiles(reverse[0])[:,1])),scope='Train numerical probe; no Validation or predictive gain claim.')
 write(OUT/'diagnostico_orden_atencion.json',result);print(json.dumps(result),flush=True)
"""
ast.parse(prefix);(root/'sondar_orden_atencion_train.py').write_text(prefix,encoding='utf-8')
print('Original failed Train proof archived; probability/median probe prepared.')
