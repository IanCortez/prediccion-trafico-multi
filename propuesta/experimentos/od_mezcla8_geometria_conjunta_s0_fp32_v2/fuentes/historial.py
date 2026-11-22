from common import *
from features import features
DEFS=[
("O1000","global",30),("D1000","global",30),("OD1000","global",20),
("O500","O1000",30),("D500","D1000",30),("OD500","OD1000",20),("OD250","OD500",20),
("O500how","O500",20),("D500how","D500",20),
("OD1000time","OD1000",10),("OD500time","OD500",10),("OD250time","OD250",15),
("OD1000how","OD1000",20)]
def keys(d):
    f,_=features(d);out={}
    for cell,tag in ((1.,"1000"),(.5,"500"),(.25,"250")):
        k=int(20/cell);C=k*k
        def encode(x,y):
            xx=np.clip(np.floor((x+10)/cell).astype(np.int64),0,k-1)
            yy=np.clip(np.floor((y+10)/cell).astype(np.int64),0,k-1)
            return xx*k+yy
        o=encode(f[:,0],f[:,1]);dd=encode(f[:,2],f[:,3]);out["O"+tag]=o;out["D"+tag]=dd;out["OD"+tag]=o*C+dd
    how=f[:,15].astype(np.int64)*24+np.floor(f[:,14]).astype(np.int64)
    typ=(f[:,15]>=5).astype(np.int64);time=typ*8+(f[:,14]//3).astype(np.int64)
    for name in ("O500","D500","OD1000"):out[name+"how"]=out[name]*168+how
    for name in ("OD1000","OD500","OD250"):out[name+"time"]=out[name]*16+time
    return out
def physical(d):return 60+np.asarray(d["od_m"],dtype=np.float64)/(20/3.6)
class HistoricalEncoder:
    def fit(self,k,y,base,mask=None):
        mask=np.arange(len(y)) if mask is None else mask
        r=np.log(y[mask]/base[mask]);self.global_mean=float(r.mean());self.global_var=float(r.var());self.tables={}
        for name,parent,shrink in DEFS:
            unique,inv=np.unique(k[name][mask],return_inverse=True)
            count=np.bincount(inv);mean=np.bincount(inv,weights=r)/count
            var=np.maximum(np.bincount(inv,weights=r*r)/count-mean*mean,0)
            self.tables[name]=(unique,count,mean,var)
        return self
    def transform(self,k,base):
        n=len(base);means={"global":np.full(n,self.global_mean)};variances={"global":np.full(n,self.global_var)}
        cols=[];names=[]
        for name,parent,shrink in DEFS:
            unique,count,mean,var=self.tables[name];pos=np.searchsorted(unique,k[name]);safe=np.minimum(pos,len(unique)-1)
            valid=(pos<len(unique))&(unique[safe]==k[name])
            nn=np.where(valid,count[safe],0);mm=np.where(valid,mean[safe],0);vv=np.where(valid,var[safe],0)
            prior=means[parent];pv=variances[parent];m=(nn*mm+shrink*prior)/(nn+shrink)
            variance=(nn*(vv+(mm-m)**2)+shrink*(pv+(prior-m)**2))/(nn+shrink)
            means[name]=m;variances[name]=variance
            cols.extend([m,np.log1p(nn),np.sqrt(variance),base*np.exp(m)/600])
            names.extend([name+"_logpace",name+"_logcount",name+"_logpace_std",name+"_estimate_600s"])
        return np.column_stack(cols).astype(np.float32),names

