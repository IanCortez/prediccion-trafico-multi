"""Independent public Europe/Lisbon timezone verification, actual model method; CPU only."""
from base_replanteo import *
import torch,types,math
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from importlib.resources import files
from datos_progreso_recurrente import lisbon_transitions
from modelo_progreso_recurrente import ProgressNet
access=deny_test();zone=ZoneInfo('Europe/Lisbon')
start,end=lisbon_transitions();ctx=types.SimpleNamespace(
 summer_start=torch.as_tensor(start,dtype=torch.float64),summer_end=torch.as_tensor(end,dtype=torch.float64))
train=np.load(DATA/'train_x.npz')['timestamp'];rg=np.random.default_rng(1742)
timestamps=np.r_[train[rg.choice(len(train),1000,replace=False)].astype(np.float64),
 np.ravel(np.r_[start,end][:,None]+np.asarray([-3601,-1,-.123456,0,.123456,1,3601])[None,:])]
actual=ProgressNet.calendar(ctx,torch.as_tensor(timestamps,dtype=torch.float64)).numpy()
expected=[]
for t in timestamps:
 dt=datetime.fromtimestamp(float(t),zone);seconds=dt.hour*3600+dt.minute*60+dt.second+dt.microsecond/1e6
 day=seconds/86400;week=dt.weekday()/7;hour=(dt.minute*60+dt.second+dt.microsecond/1e6)/3600
 expected.append([math.sin(2*math.pi*day),math.cos(2*math.pi*day),
  math.sin(2*math.pi*week),math.cos(2*math.pi*week),math.sin(2*math.pi*hour),math.cos(2*math.pi*hour)])
error=float(np.abs(actual-np.asarray(expected,dtype=np.float32)).max());assert error<2e-6
zonefile=Path(str(files('tzdata').joinpath('zoneinfo','Europe','Lisbon'))).resolve();assert zonefile.is_relative_to(ROOT.resolve())
result=dict(passed=True,timestamps=len(timestamps),actual_model_calendar_method=True,CPU_only=True,
 includes_DST_transition_boundaries_and_fractional_seconds=True,max_calendar_feature_error=error,
 timezone_file=str(zonefile.relative_to(ROOT)),timezone_file_sha256=sha(zonefile),
 model_source_sha256=sha(OUT/'modelo_progreso_recurrente.py'),source_sha256=sha(__file__),no_Validationlabels_or_Test=True)
write(OUT/'verificacion_calendario_progreso_recurrente.json',result);print(json.dumps(result),flush=True)
