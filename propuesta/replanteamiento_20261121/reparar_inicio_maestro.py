from base_replanteo import *
import shutil
p=OUT/'piloto_maestro_probabilistico.py'
assert not (OUT/'piloto_maestro_probabilistico_s0').exists(),'No restart of a partially trained run'
shutil.copy2(p,OUT/'piloto_maestro_probabilistico.inicio_fallido.txt')
shutil.copy2(OUT/'piloto_maestro_probabilistico_s0.log',OUT/'piloto_maestro_probabilistico.inicio_fallido.log')
lines=p.read_text(encoding='utf-8-sig').splitlines()
index=[i for i,l in enumerate(lines) if 'nll=-np.log(prob[' in l];assert len(index)==1
lines[index[0]]=" nll=-np.log(prob[np.arange(len(truth)),label]);crps=15*np.square(cdf-(grid[None,:]>=truth[:,None])).sum(1)"
s='\n'.join(lines)+'\n';compile(s,str(p),'exec');p.write_text(s,encoding='utf-8')
write(OUT/'recuperacion_inicio_maestro.json',dict(original_exit_code=1,stage='Syntax only; no run directory, data read, model fit or parameter update occurred',fix='Use NumPy square directly for discrete CRPS; remove malformed unused branch',GPU_untouched=True,source_sha256=sha(p)))
print('Arranque corregido; no se repite ningún entrenamiento previo')
