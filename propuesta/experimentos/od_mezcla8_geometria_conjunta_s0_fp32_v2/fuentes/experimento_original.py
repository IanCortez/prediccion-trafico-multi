from common import *
from features import *
import argparse
p=argparse.ArgumentParser();p.add_argument("--id",required=True);p.add_argument("--seed",type=int,default=0);p.add_argument("--pace",action="store_true");p.add_argument("--pos",action="store_true")
a=p.parse_args();run=RUNS/a.id;run.mkdir(exist_ok=False);snapshot(run,vars(a));started=time.time()
sys.path.insert(0,str(ROOT/"codigo-ian/src"))
import trajectory_loader as tl,urban_context as uc,road_network as RN,tte_model as M
import torch,random
torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
trips=tl.load_trips(ROOT/"auditoria/muestras/porto/central_6000.csv","porto",verbose=False)
ctx=uc.UrbanContext(pois=ROOT/"tesis/osm/porto_pois.geojson",road_nodes=ROOT/"tesis/osm/porto_roads.geojson",city="porto",verbose=False)
city=tl.CityTrips(trips,"porto",cell_m=1500,urban_ctx=ctx,road_graph=RN.load_graph(ROOT/"tesis/osm","porto"),verbose=False)
cd=M.build_city_data(city,target_fraction=.1,posenc=M.POS_SCALES_KM if a.pos else ())
model=M.TravelTimeModel(dim=64,hidden=128,dropout=.1,bank_size=max(cd.n_cells//2,8),pace=a.pace,n_extra=len(M.POSENC_NAMES) if a.pos else 0)
trainer=M.Trainer(model,cd,[],device="cuda",batch_size=512)
trainer.train(epochs=100,patience=15)
vp=trainer.predict(cd,cd.val_idx);v=cd.raw_times[cd.val_idx].numpy()
trainer.save(run/"modelo.pt")
np.savez_compressed(run/"validacion.npz",row=np.asarray(cd.val_idx),pred=vp,true=v,timestamp=np.array([city.trips[i]["timestamp"] for i in cd.val_idx]))
rec=dict(id=a.id,architecture="upstream original diagnostic",config=vars(a),cohort="previous central6000 diagnostic, 480 train/600 validation; not primary full cohort",validation=metrics(v,vp),train_n=len(cd.train_idx),elapsed_seconds=time.time()-started,conclusion="Diagnóstico de la opción pace ya existente. No es comparable numéricamente con las corridas de 1.023.529 viajes de entrenamiento.")
write_json(run/"resultado.json",rec);register(rec);print(json.dumps(rec),flush=True)

