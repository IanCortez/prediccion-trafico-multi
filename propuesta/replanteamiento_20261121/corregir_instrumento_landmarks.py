from base_replanteo import *
import ast
old=OUT/'landmarks_red_publica.py';s=old.read_text()
s=s.replace("np.broadcast_to(-.5*(error/30)**2,values.shape)","np.broadcast_to((-.5*(error/30)**2)[...,None],values.shape)")
p=OUT/'landmarks_red_publica_v2.py';assert not p.exists();ast.parse(s);p.write_text(s)
s=(OUT/'piloto_landmarks_red_Train.py').read_text()
s=s.replace('from landmarks_red_publica import','from landmarks_red_publica_v2 import')
s=s.replace("'piloto_landmarks_red_Train_s0'","'piloto_landmarks_red_Train_v2_s0'")
s=s.replace("'piloto_landmarks_red_Train.py'","'piloto_landmarks_red_Train_v2.py'").replace("'landmarks_red_publica.py'","'landmarks_red_publica_v2.py'")
p=OUT/'piloto_landmarks_red_Train_v2.py';assert not p.exists();ast.parse(s);p.write_text(s)
write(OUT/'piloto_landmarks_red_Train_s0/interrupcion_instrumento.json',dict(
 scientific_model_fit=False,scientific_score_computed=False,
 error='Projectionweights lacked singletonlandmarkdimension for512x4x32broadcast.',
 correction='Addexplicitfinalaxis, samegeometry/weights/landmarks/rows/targets/settings.',
 next_run='piloto_landmarks_red_Train_v2_s0',original_frozen_sources_preserved=True))
print('Projectionweightdimensions corrected before anymodel fit; originalattempt frozen.')
