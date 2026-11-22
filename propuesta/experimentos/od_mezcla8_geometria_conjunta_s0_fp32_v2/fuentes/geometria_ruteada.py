from common import *
from numba import njit,prange

@njit(cache=True,parallel=True)
def sample_routes(rawxy,flat,offsets,prefix,junctions,indptr,indices,chosen,pred,lookup,choice,oc,dc,op,dp,oj,dj,orig,dest):
    n=len(choice);out=np.zeros((n,21),np.float32);nj=len(junctions)
    for qi in prange(n):
        c=choice[qi]
        if c<0:continue
        os=(c%4)//2;ds=c%2
        # Construct chain segments in reverse route order, then visit forwards.
        chains=np.empty(nj+2,np.int32);starts=np.empty(nj+2,np.int32);ends=np.empty(nj+2,np.int32);ns=0
        if c>=4:
            chain=oc[qi,os];chains[0]=chain;starts[0]=op[qi,os];ends[0]=dp[qi,ds];ns=1
        else:
            chain=dc[qi,ds]
            if chain>=0:
                chains[ns]=chain;starts[ns]=offsets[chain];ends[ns]=dp[qi,ds];ns+=1
            node=dj[qi,ds];root=oj[qi,os];sr=lookup[root]
            valid=True
            while node!=root:
                parent=pred[sr,node]
                if parent<0 or ns>=nj:valid=False;break
                edge=-1
                for k in range(indptr[parent],indptr[parent+1]):
                    if indices[k]==node:edge=k;break
                if edge<0:valid=False;break
                chain=chosen[edge];chains[ns]=chain;starts[ns]=offsets[chain];ends[ns]=offsets[chain+1]-1;ns+=1;node=parent
            if not valid:continue
            chain=oc[qi,os]
            if chain>=0:
                chains[ns]=chain;starts[ns]=op[qi,os];ends[ns]=offsets[chain+1]-1;ns+=1
        count=3
        for si in range(ns):count+=ends[si]-starts[si]+1
        pts=np.empty((count,2),np.float64);pts[0]=orig[qi];m=1
        for si in range(ns-1,-1,-1):
            for pos in range(starts[si],ends[si]+1):
                p=rawxy[flat[pos]]
                if (p[0]-pts[m-1,0])**2+(p[1]-pts[m-1,1])**2>1e-12:pts[m]=p;m+=1
        if ns==0:
            # Identical snapped junctions can yield an empty chain route.
            p=rawxy[junctions[oj[qi,os]]]
            if (p[0]-pts[m-1,0])**2+(p[1]-pts[m-1,1])**2>1e-12:pts[m]=p;m+=1
        pts[m]=dest[qi];m+=1
        cum=np.zeros(m,np.float64)
        angles=0.;signed=0.;turns=0.;maxangle=0.;lastx=0.;lasty=0.
        for i in range(1,m):
            vx=pts[i,0]-pts[i-1,0];vy=pts[i,1]-pts[i-1,1];length=(vx*vx+vy*vy)**.5
            cum[i]=cum[i-1]+length
            if length>1e-6:
                if lastx*lastx+lasty*lasty>1e-12:
                    angle=np.arctan2(lastx*vy-lasty*vx,lastx*vx+lasty*vy)
                    angles+=abs(angle);signed+=angle;turns+=abs(angle)>np.pi/4;maxangle=max(maxangle,abs(angle))
                lastx=vx;lasty=vy
        total=cum[-1];od=np.sqrt(((dest[qi]-orig[qi])**2).sum());cursor=1
        for j in range(1,8):
            fraction=j/8.;target=total*fraction
            while cursor<m-1 and cum[cursor]<target:cursor+=1
            alpha=(target-cum[cursor-1])/max(cum[cursor]-cum[cursor-1],1e-9)
            point=pts[cursor-1]*(1-alpha)+pts[cursor]*alpha
            chord=orig[qi]*(1-fraction)+dest[qi]*fraction
            out[qi,2*(j-1):2*j]=(point-chord)/1000
        out[qi,14]=total/1000;out[qi,15]=np.log(max(total/max(od,1),1))
        out[qi,16]=angles/np.pi;out[qi,17]=signed/np.pi;out[qi,18]=turns;out[qi,19]=maxangle/np.pi;out[qi,20]=1
    return out

