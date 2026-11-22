from common import *
from features import *
from calendario_porto import *
import pandas as pd
# Historical rules and local-date behavior, independent of travel-time labels.
assert date(2013,10,5) not in special_dates(2013);assert date(2014,11,1) not in special_dates(2014);assert date(2014,12,1) not in special_dates(2014);assert date(2014,6,19) not in special_dates(2014)
assert date(2014,4,25) in special_dates(2014) and date(2014,5,1) in special_dates(2014) and date(2014,6,24) in special_dates(2014)
query=pd.to_datetime(['2014-04-30T23:30:00Z','2014-05-02T12:00:00Z'],utc=True).tz_convert('Europe/Lisbon');cols=calendar_columns(query);assert cols['calendar_special'][0]==1 and cols['calendar_bridge'][1]==1
va=dict(np.load(DATA/'validation_x.npz'));X,names=features(va,calendar=True);flags=X[:,names.index('calendar_special')]>0;z=np.load(RUNS/'od_tabm_geometria_rutas_s0_fp32_v2/validacion.npz');slices={}
for name,mask in (('special_day',flags),('ordinary_day',~flags)):
    slices[name]=dict(n=int(mask.sum()),mae_s=float(np.abs(z['pred'][mask]-z['true'][mask]).mean()),bias_s=float(np.mean(z['pred'][mask]-z['true'][mask])))
write_json(BASE/'instrumentos/calendario_porto.json',dict(passed=True,sources=SOURCES,calendar='Frozen 2013-2014; national, Porto June24 and Carnival activity indicators',historical_suspension_verified=True,timezone_boundary_verified=True,slices_validation=slices,test_labels_read=False))
print(slices,flush=True)
