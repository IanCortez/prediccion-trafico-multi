from base_replanteo import *
s=(OUT/'entrenar_espacial.py').read_text(encoding='utf-8-sig')
s=s.replace('from modelo_espacial import SpatialArrival','from modelo_direccional import DirectionalArrival,heading_ids').replace('model=SpatialArrival(','model=DirectionalArrival(')
s=s.replace('dp_batch=4,inference_batch=32','dp_batch=2,inference_batch=16').replace("geometry='200m cells; public2014 roads and Train transitions only'","geometry='200m cells; public2014 roads and Train transitions only; state=(cell,incoming heading4) plus one unknown-heading origin state'")
s=s.replace("['entrenar_espacial.py','modelo_espacial.py','nucleo_semi_markov.py','base_replanteo.py']","['entrenar_direccional.py','modelo_direccional.py','nucleo_direccional.py','modelo_espacial.py','nucleo_semi_markov.py','base_replanteo.py']")
old="p,wm=model.state_parameters(q,x[ix,:4],safe,dd);dx="
new="previous=cells[(ri-1).clamp_min(0)].long();has_previous=(ri>offset[ti])&(previous<model.N)&valid[(ri-1).clamp_min(0)];pdx=torch.div(cur,grid['ny'],rounding_mode='floor')-torch.div(previous,grid['ny'],rounding_mode='floor');pdy=cur%grid['ny']-previous%grid['ny'];heading=torch.where(has_previous,heading_ids(pdx,pdy),4);p,wm=model.state_parameters(q,x[ix,:4],safe,dd,heading);e,we=model.end_parameters(e,we,heading);dx="
assert old in s;s=s.replace(old,new)
s=s.replace('range(0,len(use),32)','range(0,len(use),16)').replace('use[first:first+32]','use[first:first+16]').replace('if n<32:','if n<16:').replace('expand(32-n)','expand(16-n)').replace('torch.randint(len(trip),(4,)','torch.randint(len(trip),(2,)')
s=s.replace("sha(OUT/'modelo_espacial.py')","sha(OUT/'modelo_direccional.py')").replace("sha(OUT/'nucleo_semi_markov.py')","sha(OUT/'nucleo_direccional.py')")
(OUT/'entrenar_direccional.py').write_text(s,encoding='utf-8')
print('Separate trainer created; running spatial source unchanged.')

