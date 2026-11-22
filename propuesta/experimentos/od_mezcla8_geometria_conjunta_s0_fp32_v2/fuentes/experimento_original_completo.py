from common import *
from features import *
import argparse,torch,gc,types
p=argparse.ArgumentParser();p.add_argument("--id",required=True);p.add_argument("--seed",type=int,default=0);p.add_argument("--epochs",type=int,default=40);p.add_argument("--batch",type=int,default=1024);p.add_argument("--no-invariance",action="store_true")
a=p.parse_args();run=RUNS/a.id;run.mkdir(exist_ok=False);snapshot(run,vars(a));started=time.time()
sys.path.insert(0,str(ROOT/"codigo-ian/src"))
import trajectory_loader as tl,urban_context as uc,road_network as RN,tte_model as M
torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed)
cache=DATA/"original_completo_cd_v1.pt"
if cache.exists():
    cd=torch.load(cache,map_location="cpu",weights_only=False)
else:
    tr,y,va,v=load_development();N=len(y);nv=len(v)
    d={k:np.concatenate([tr[k],va[k]]) for k in tr};target=np.r_[y,v]
    # Trip dictionaries are adapter inputs, not observed path geometry features.
    geo=np.load(DATA/"geometria_train.npz");lookup=dict(zip(geo["row"].tolist(),geo["X"][:,0].tolist()))
    vgeo=np.load(DATA/"geometria_validation.npz");lookup.update(zip(vgeo["row"].tolist(),vgeo["X"][:,0].tolist()))
    trips=[]
    for i in range(len(target)):
        trips.append(dict(timestamp=int(d["timestamp"][i]),o_lat=float(d["o_lat"][i]),o_lon=float(d["o_lon"][i]),d_lat=float(d["d_lat"][i]),d_lon=float(d["d_lon"][i]),od_dist=float(d["od_m"][i]),travel_time=float(target[i]),call_type="C",day_type="A",route_len=float(lookup.get(int(d["row"][i]),float("nan")))))
        if i%250000==0:print(dict(stage="trips",n=i,seconds=time.time()-started),flush=True)
    # Fixed geographic cells, independent of held-out coordinates.
    class FixedGrid(tl.RegionGrid):
        def fit(self,lats,lons,quantile=0):
            self.lat0,self.lat_max,self.lon0,self.lon_max=41.1405,41.1865,-8.6887,-8.5557
            mid=(self.lat0+self.lat_max)/2
            self._dlat=self.cell_m/110540;self._dlon=self.cell_m/(111320*np.cos(np.deg2rad(mid)))
            ni=int((self.lat_max-self.lat0)//self._dlat)+1;nj=int((self.lon_max-self.lon0)//self._dlon)+1
            self.cell_ids=[(i,j) for i in range(ni) for j in range(nj)]
            self.cell_index={k:i for i,k in enumerate(self.cell_ids)}
            return self
    tl.RegionGrid=FixedGrid
    from scipy.stats import rankdata
    uc.quantile_rank=lambda values:(rankdata(values,method="average")-1)/max(len(values)-1,1)
    def fixed_corridors(items):
        ref=np.sort(np.array([t.get("_poi_raw",0) for t in items[:N]],dtype=float))
        values=np.array([t.get("_poi_raw",0) for t in items],dtype=float)
        ranks=(np.searchsorted(ref,values,side="left")+np.searchsorted(ref,values,side="right")-1)/2
        for t,q in zip(items,np.clip(ranks/max(len(ref)-1,1),0,1)):
            t["poi_corridor_q"]=float(q);t.pop("_poi_raw",None)
    uc.finalize_edge_context=fixed_corridors
    ctx=uc.UrbanContext(pois=ROOT/"tesis/osm/porto_pois.geojson",road_nodes=ROOT/"tesis/osm/porto_roads.geojson",city="porto",verbose=False)
    city=tl.CityTrips(trips,"porto",cell_m=500,urban_ctx=ctx,road_graph=RN.load_graph(ROOT/"tesis/osm","porto"),verbose=True)
    print(dict(stage="city_ready",cells=city.grid.n_cells,seconds=time.time()-started),flush=True)
    city.chronological_split=types.MethodType(lambda self,**kw:(list(range(N)),list(range(N,N+nv)),[]),city)
    cd=M.build_city_data(city,posenc=M.POS_SCALES_KM)
    torch.save(cd,cache)
    write_json(DATA/"original_completo_cache.json",dict(n_train=N,n_validation=nv,n_test=0,cell_m=500,n_cells=cd.n_cells,call_type="constant C for OD-only parity",fixes=["fixed predefined geographic grid","average tied static percentiles","corridor empirical CDF from training inputs only"],original_core="unchanged neural architecture and losses; pace and Fourier positions enabled",manifest_sha256=sha(DATA/"manifiesto.json"),cache_sha256=sha(cache),seconds=time.time()-started))
    del city,trips,lookup,d,tr,va,ctx;gc.collect()
model=M.TravelTimeModel(dim=64,hidden=128,dropout=.1,bank_size=max(cd.n_cells//2,8),pace=True,n_extra=len(M.POSENC_NAMES))
trainer=M.Trainer(model,cd,[],device="cuda",batch_size=a.batch,val_size=len(cd.val_idx),alpha=0 if a.no_invariance else .5,beta=0 if a.no_invariance else .5)
trainer.train(epochs=a.epochs,patience=10)
pred=trainer.predict(cd,cd.val_idx);truth=cd.raw_times[cd.val_idx].cpu().numpy()
trainer.save(run/"modelo.pt")
rows=np.load(DATA/"validation_x.npz")
np.savez_compressed(run/"validacion.npz",row=rows["row"],timestamp=rows["timestamp"],pred=pred,true=truth)
rec=dict(id=a.id,architecture="upstream full fixed-contract",config=vars(a),validation=metrics(truth,pred),interval=block_interval(truth,pred,rows["timestamp"]),train_n=len(cd.train_idx),parameters=sum(z.numel() for z in model.parameters()),epochs=len(trainer.history["train"]),elapsed_seconds=time.time()-started,model_sha256=sha(run/"modelo.pt"),conclusion="Comparador del modelo original con las mismas etiquetas y validación completa; solo correcciones de contrato y opción pace/posiciones ya existentes. Sin ciudades fuente reales.")
write_json(run/"resultado.json",rec);register(rec);print(json.dumps(rec),flush=True)

