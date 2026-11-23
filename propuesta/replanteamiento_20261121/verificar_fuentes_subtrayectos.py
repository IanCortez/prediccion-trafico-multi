from base_replanteo import *
import ast
names=['tokens_subtrayectos.py','normalizar_tokens_subtrayectos.py','modelo_atencion_subtrayectos.py','entrenar_atencion_subtrayectos.py']
for name in names:ast.parse((OUT/name).read_text(),filename=name)
write(OUT/'fuentes_subtrayectos_verificadas.json',dict(syntax_valid=True,sources={name:sha(OUT/name) for name in names},runtime_proofs='Will run on actualTrain and raw5queries before acceptance.'))
print('Syntax valid for all four newlyprepared sources.')
