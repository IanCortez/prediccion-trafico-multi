from common import *
import pandas as pd
# Geographic origin, unit conversion, and feature scales are fixed before fitting.
ORIGIN=(-8.62,41.16)
def features(d,enriched=False,fourier=False,calendar=False):
    x0=(d["o_lon"]-ORIGIN[0])*111.195*np.cos(np.deg2rad(ORIGIN[1]))
    y0=(d["o_lat"]-ORIGIN[1])*111.195
    x1=(d["d_lon"]-ORIGIN[0])*111.195*np.cos(np.deg2rad(ORIGIN[1]))
    y1=(d["d_lat"]-ORIGIN[1])*111.195
    dx=x1-x0;dy=y1-y0;dist=np.maximum(d["od_m"]/1000,1e-4)
    dt=pd.to_datetime(d["timestamp"],unit="s",utc=True).tz_convert("Europe/Lisbon")
    hour=np.asarray(dt.hour+dt.minute/60+dt.second/3600);dow=np.asarray(dt.dayofweek)
    cols=dict(ox=x0,oy=y0,dx=x1,dy=y1,delta_x=dx,delta_y=dy,od_km=dist,log_od=np.log(dist),abs_x=np.abs(dx),abs_y=np.abs(dy),bearing_x=dx/dist,bearing_y=dy/dist,mid_x=(x0+x1)/2,mid_y=(y0+y1)/2,hour=hour,dow=dow,weekend=(dow>=5).astype(float),month=np.asarray(dt.month))
    for period,v,name in ((24,hour,"hour"),(7,dow+hour/24,"week"),(12,np.asarray(dt.month)-1,"season")):
        cols[name+"_sin"]=np.sin(2*np.pi*v/period);cols[name+"_cos"]=np.cos(2*np.pi*v/period)
    for angle in (np.pi/4,-np.pi/4):
        for tag,x,y in (("o",x0,y0),("d",x1,y1)):
            cols[f"{tag}_rot{angle:.2f}a"]=x*np.cos(angle)+y*np.sin(angle)
            cols[f"{tag}_rot{angle:.2f}b"]=-x*np.sin(angle)+y*np.cos(angle)
    if fourier:
        for tag,v in (("ox",x0),("oy",y0),("dx",x1),("dy",y1)):
            for scale in (.1,.25,.5,1,2,4,8):
                cols[f"{tag}_sin_{scale}"]=np.sin(2*np.pi*v/scale)
                cols[f"{tag}_cos_{scale}"]=np.cos(2*np.pi*v/scale)
    if enriched:
        for c in ("A","B","C"):cols["call_"+c]=(d["call"]==c).astype(float)
    if calendar:
        from calendario_porto import calendar_columns
        cols.update(calendar_columns(dt))
    names=list(cols);X=np.column_stack(list(cols.values())).astype(np.float32)
    assert np.isfinite(X).all()
    return X,names
def load_development():
    return (dict(np.load(DATA/"train_x.npz")),np.load(DATA/"train_y.npy"),dict(np.load(DATA/"validation_x.npz")),np.load(DATA/"validation_y.npy"))
def fit_power(d,y):
    z=np.log(np.maximum(d["od_m"].astype(np.float64),1))
    beta,intercept=np.polyfit(z,np.log(y),1)
    pred=np.exp(intercept+beta*z)
    # Multiplicative median calibrated on train only.
    intercept+=np.log(np.median(y/pred))
    return float(intercept),float(beta)
def power_predict(d,coef):
    return np.exp(coef[0]+coef[1]*np.log(np.maximum(d["od_m"],1)))
def block_interval(y,p,timestamp,repeats=4000,seed=1729):
    days=np.asarray(timestamp)//86400
    _,inv=np.unique(days,return_inverse=True)
    sums=np.bincount(inv,weights=np.abs(np.asarray(y)-p));counts=np.bincount(inv)
    rng=np.random.default_rng(seed);ix=rng.integers(0,len(sums),(repeats,len(sums)))
    means=sums[ix].sum(1)/counts[ix].sum(1)
    return dict(ci95_s=np.quantile(means,[.025,.975]).tolist(),blocks=len(sums),method="bootstrap departure UTC day; conditional on fitted model")
def diagnose(d,y,p):
    e=np.abs(y-p);out={}
    for name,v,edges in (("distance_km",d["od_m"]/1000,[0,1,2,4,8,1000]),("hour_local",features(d)[0][:,14],[0,6,10,16,20,24])):
        rows=[]
        for a,b in zip(edges[:-1],edges[1:]):
            m=(v>=a)&(v<b)
            if m.any():rows.append(dict(interval=[a,b],n=int(m.sum()),mae_s=float(e[m].mean()),bias_s=float((p[m]-y[m]).mean())))
        out[name]=rows
    return out



