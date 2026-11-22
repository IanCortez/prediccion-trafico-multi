from common import *
from features import ORIGIN
from numba import njit
import csv
@njit(cache=True)
def summarize(xy,offsets):
    n=len(offsets)-1;out=np.zeros((n,3),np.float32);maximum=np.zeros(n,np.float32)
    for i in range(n):
        for j in range(offsets[i],offsets[i+1]-1):
            d=np.sqrt(((xy[j+1]-xy[j])**2).sum())*1000;maximum[i]=max(maximum[i],d)
            for k,threshold in enumerate((2.,5.,10.)):
                if d<threshold:out[i,k]+=15
    return out,maximum
if __name__=='__main__':
    started=time.time();z=dict(np.load(DATA/'trayectorias_train.npz'));stop,maximum=summarize(z['xy'],z['offsets']);duration=np.diff(z['offsets'])*15-15;assert np.all(stop<=duration[:,None]);path=DATA/'pausas_gps_train.npz';assert not path.exists();np.savez_compressed(path,row=z['row'],low_displacement_seconds=stop,reliable=maximum<=500,threshold_m=np.array([2.,5.,10.]));del z
    va=dict(np.load(DATA/'validation_x.npz'));y=np.load(DATA/'validation_y.npy');lookup={int(row):i for i,row in enumerate(va['row'])};out=np.zeros((len(y),3),np.float32);maxval=np.zeros(len(y),np.float32);seen=np.zeros(len(y),bool);csv.field_size_limit(2**31-1)
    with (ROOT/'tesis/porto/porto/train.csv').open(encoding='utf-8-sig',newline='') as f:
        for row,r in enumerate(csv.DictReader(f)):
            i=lookup.get(row)
            if i is None:continue
            points=np.array(json.loads(r['POLYLINE']));assert (len(points)-1)*15==y[i];xy=np.column_stack([(points[:,0]-ORIGIN[0])*111.195*np.cos(np.deg2rad(ORIGIN[1])),(points[:,1]-ORIGIN[1])*111.195]);s,m=summarize(xy,np.array([0,len(xy)],np.int64));out[i]=s[0];maxval[i]=m[0];seen[i]=True
    assert seen.all();validation_path=BASE/'instrumentos/diagnostico_pausas_validation.npz';np.savez_compressed(validation_path,row=va['row'],low_displacement_seconds=out,reliable=maxval<=500,threshold_m=np.array([2.,5.,10.]));record=dict(train_n=len(stop),validation_n=len(y),threshold_m=[2,5,10],train_mean_low_displacement_seconds=stop.mean(0),validation_mean_low_displacement_seconds=out.mean(0),train_reliable_n=int((maximum<=500).sum()),validation_reliable_n=int((maxval<=500).sum()),source_train_sha256=sha(DATA/'trayectorias_train.npz'),source_csv_sha256=json.loads((DATA/'manifiesto.json').read_text())['source_sha256'],output_hashes={str(p.relative_to(ROOT)):sha(p) for p in (path,validation_path)},usage='Training auxiliary targets and privileged validation diagnostics only. Low displacement is an operational GPS proxy, not annotated stationary/parking time. Never an OD query input.',test_labels_read=False,test_path_decoded=False,seconds=time.time()-started);write_json(DATA/'pausas_gps_manifiesto.json',record);print(json.dumps(record,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)),flush=True)
