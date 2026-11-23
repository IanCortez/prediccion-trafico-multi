"""Train OD feature construction with whole-query-week historical exclusion."""
import numpy as np
from consulta_od import od_records,build_od_features
from consulta_historicos_cruzados import prepare,history_features,STATIC_NAMES,MOVEMENT_NAMES
def build_training_features(items,config):
 # Public-like records have exactly five fields; provenance and targets never enter.
 d=od_records(items)
 base_config=dict(config);base_config.update(projected_routes=False,mapmatched=False,learned_route=False,road_attributes=False)
 X,names=build_od_features(d,base_config)
 p=prepare(items);fold=(d['timestamp']//(7*86400))%5
 movement=np.empty((len(items),10),np.float32);assigned=np.zeros(len(items),np.uint8);static=None
 for f in range(5):
  M,S=history_features(p,f);selected=fold==f;movement[selected]=M[selected];assigned[selected]+=1
  if static is None:static=S
  else:assert np.array_equal(static,S)
 assert np.all(assigned==1)
 return np.column_stack([X,static,movement]).astype(np.float32),names+STATIC_NAMES+MOVEMENT_NAMES
