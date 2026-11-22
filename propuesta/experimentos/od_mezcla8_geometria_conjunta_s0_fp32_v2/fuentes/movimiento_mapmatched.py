from common import *
from map_matching import node_connections,edge_distance
from trazado_vial import collect_edge_path
from ruteo_compacto import compressed_csr
from numba import njit,prange,set_num_threads
import pandas as pd

@njit(cache=True,parallel=True)
def accumulate_motion(xy,offsets,timestamp,fold,matched,fraction,error,u,v,length,network_D,conn,routing,transitions,utc_offsets):
    distance=np.zeros((5,168,len(u)),np.float32);seconds=np.zeros_like(distance);counts=np.zeros((5,7),np.int64);totals=np.zeros(5,np.float64);max_error=np.zeros(5,np.float64)
    for f in prange(5):
        ids=np.empty(4096,np.int32);parts=np.empty(4096,np.float64)
        for trip in range(len(timestamp)):
            if fold[trip]!=f:continue
            start,end=offsets[trip],offsets[trip+1]
            for point in range(start,end-1):
                dx=(xy[point+1,0]-xy[point,0])*1000;dy=(xy[point+1,1]-xy[point,1])*1000;gps=np.sqrt(dx*dx+dy*dy)
                if gps>500:counts[f,2]+=1;continue
                if error[point]>80 or error[point+1]>80:counts[f,3]+=1;continue
                ea=matched[point];eb=matched[point+1];fa=fraction[point];fb=fraction[point+1]
                instant=timestamp[trip]+(point-start)*15+7;segment=np.searchsorted(transitions,instant,side='right')-1;wall=instant+utc_offsets[segment];how=((wall//86400+3)%7)*24+(wall%86400)//3600
                if gps<5:
                    seconds[f,how,ea]+=15;counts[f,1]+=1;continue
                network=edge_distance(ea,fa,eb,fb,u,v,length,network_D,*conn)
                if not np.isfinite(network) or network>750:counts[f,4]+=1;continue
                if network<5:
                    seconds[f,how,ea]+=15;counts[f,1]+=1;continue
                used,expected=collect_edge_path(ea,fa,eb,fb,u,v,length,network_D,conn,routing,ids,parts)
                if used<0:counts[f,5]+=1;continue
                total=0.
                for k in range(used):total+=parts[k]
                difference=abs(total-expected)
                if difference>.25 or total<=0:counts[f,6]+=1;continue
                max_error[f]=max(max_error[f],difference)
                for k in range(used):
                    edge=ids[k];distance[f,how,edge]+=parts[k];seconds[f,how,edge]+=15*parts[k]/total
                counts[f,0]+=1;totals[f]+=total
    return distance,seconds,counts,totals,max_error

if __name__=='__main__':
    started=time.time();set_num_threads(5);folder=DATA/'movimiento_mapmatched_2014';folder.mkdir(exist_ok=True);assert not (folder/'manifiesto.json').exists()
    assert json.loads((BASE/'instrumentos/trazado_vial.json').read_text())['passed'];z=dict(np.load(DATA/'trayectorias_train.npz'));match=DATA/'map_matching_2014_v2';meta=json.loads((match/'manifiesto.json').read_text());assert meta['config']['source_sha256']==sha(DATA/'trayectorias_train.npz')
    raw=dict(np.load(DATA/'mapa_dirigido_2014.npz'));comp=dict(np.load(DATA/'grafo_compacto_2014.npz'));D=np.load(DATA/'distancias_2014/D.npy',mmap_mode='r');conn=node_connections(comp);graph,chosen=compressed_csr(comp,'length');routing=(comp['chain_offsets'],comp['flat_incoming_arc'],graph.indptr,graph.indices,chosen,comp['chain_length'])
    ts=z['timestamp'];low=pd.Timestamp(int(ts.min()),unit='s',tz='UTC').floor('D');high=pd.Timestamp(int(ts.max())+1800,unit='s',tz='UTC').ceil('D');hours=pd.date_range(low,high,freq='h');utc=hours.as_unit('s').asi8;wall=hours.tz_convert('Europe/Lisbon').tz_localize(None).as_unit('s').asi8;offset=wall-utc;starts=np.r_[0,np.flatnonzero(np.diff(offset)!=0)+1];transitions=utc[starts];utc_offsets=offset[starts];fold=(ts//(7*86400))%5
    outputs=accumulate_motion(z['xy'],z['offsets'],ts,fold,np.load(match/'edge.npy',mmap_mode='r'),np.load(match/'fraction.npy',mmap_mode='r'),np.load(match/'error_m.npy',mmap_mode='r'),raw['u'],raw['v'],raw['length'],D,conn,routing,transitions,utc_offsets)
    dist,sec,counts,totals,max_error=outputs;accepted=(counts[:,:2].sum(1)*15).astype(float);actual_time=sec.sum((1,2),dtype=np.float64);actual_length=dist.sum((1,2),dtype=np.float64)
    assert np.all(np.abs(actual_time-accepted)/accepted<1e-4);assert np.all(np.abs(actual_length-totals)/np.maximum(totals,1)<1e-4);assert counts.sum()==len(z['xy'])-len(ts)
    np.save(folder/'distance_m.npy',dist);np.save(folder/'seconds.npy',sec)
    write_json(folder/'manifiesto.json',dict(source_map_matching_sha256=sha(match/'manifiesto.json'),source_trajectories=len(ts),source_gps_points=len(z['xy']),shape=dist.shape,counts_by_fold=counts,count_names=['accepted_moving','accepted_stopped','rejected_GPS_jump','rejected_projection','rejected_network_distance','rejected_path_collection','rejected_path_length'],distance_totals_m=actual_length,time_totals_s=actual_time,expected_time_totals_s=accepted,conservation_relative_time_error=(actual_time-accepted)/accepted,conservation_relative_length_error=(actual_length-totals)/totals,max_path_error_m=max_error,clock='Exact Europe/Lisbon interval midpoint and DST transitions',folds='UTC departure week modulo 5; whole trajectories kept in one fold',assumed_GPS_interval_s=15,stops='GPS step or road displacement under 5m; 15 seconds assigned to matched directed arc',maximum_projection_error_m=80,maximum_network_transition_m=750,distance_sha256=sha(folder/'distance_m.npy'),seconds_sha256=sha(folder/'seconds.npy'),test_labels_read=False,train_only=True,code_hashes={name:sha(BASE/name) for name in ('trazado_vial.py','movimiento_mapmatched.py','map_matching.py')},seconds=time.time()-started))
    print(dict(stage='done',counts=counts.tolist(),max_path_error_m=max_error.tolist(),time_conservation_error=(actual_time-accepted).tolist(),seconds=time.time()-started),flush=True)
