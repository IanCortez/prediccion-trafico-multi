from common import *
import ast,torch
source=BASE/'referencias/TransferTraj/models/transfertraj.py';tree=ast.parse(source.read_text(encoding='utf-8-sig'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='tokenize_timestamp');scope=dict(torch=torch);exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope);temporal=torch.tensor([[[1399000000.,0.],[1399000450.,450.]]],dtype=torch.float64);tokens=scope['tokenize_timestamp'](temporal);assert float(tokens[0,1,3])==7.5;duration_true=tokens[:,1,3].numpy()*60;duration_pred=duration_true+60;assert metrics(duration_true,duration_pred)['mae_s']==60
rejected=False
try:metrics(np.array([[0.,450.]]),np.array([[0.,510.]]))
except AssertionError:rejected=True
assert rejected
write_json(BASE/'instrumentos/unidades_destino_tte.json',dict(passed=True,official_delta450s_token7_5min=True,one_minute_prediction_error_equals60s=True,known_origin_not_averaged_into_destination_error=True,multidimensional_metric_input_rejected=True,test_labels_read=False,official_source_sha256=sha(source),metric_code_sha256=sha(BASE/'common.py')));print('Destination-only units and masking contracts passed',flush=True)
