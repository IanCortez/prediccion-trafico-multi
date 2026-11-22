from common import *
for name in ('od_tabm_ensemble16_s0_fp32','od_tabm_ensemble16_s1_fp32'):
    path=RUNS/name/'resultado.json';r=json.loads(path.read_text(encoding='utf-8-sig'));backup=path.with_name('resultado_antes_de_corregir_precision.json');assert not backup.exists();backup.write_bytes(path.read_bytes())
    r['cohort']='diagnostic precision audit';r['actual_precision']='float32 with TF32 enabled';r['config']['inference_precision']='tf32';r['conclusion']='La auditoría posterior detectó TF32 habilitado al desactivar bfloat16. Se corrige la descripción de esta corrida y se conserva el resultado original como archivo separado. Esta variante diagnóstica queda excluida de la selección; la evaluación float32 estricta se repite bajo otro identificador.'
    write_json(path,r)
write_json(BASE/'instrumentos/correccion_auditoria_precision.json',dict(ids=['od_tabm_ensemble16_s0_fp32','od_tabm_ensemble16_s1_fp32'],cause='Autocast was disabled but float32 matmul precision remained high, allowing TF32. Strict float32 requires highest. Initial configuration description corrected; original result files retained.',test_labels_read=False))
