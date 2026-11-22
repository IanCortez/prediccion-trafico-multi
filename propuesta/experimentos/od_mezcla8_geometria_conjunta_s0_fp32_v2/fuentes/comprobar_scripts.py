from common import *
import ast
files=list(BASE.glob('*.py'))
for path in files:ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
print('Syntax verified:',len(files),'scripts')
for log in ('rutas_dinamicas.log','xgboost_semanal.log','xgboost_geometria.log','tabm_s1.log','tabm_geometria.log'):
    path=BASE/log
    print(log, '\n'.join(path.read_text(encoding='utf-8',errors='replace').splitlines()[-2:])[:800])
