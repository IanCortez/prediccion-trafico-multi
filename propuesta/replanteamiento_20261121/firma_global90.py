"""Geometry-only observed road fingerprint; no local times, T or GPS sample counts."""
import numpy as np
from numba import njit,prange
@njit(cache=True,parallel=True)
def geometry90(offsets,codes,distance,chord,length,free,orientation,speedbin):
    out=np.zeros((len(offsets)-1,90),np.float32)
    for i in prange(len(offsets)-1):
        total=0.;used=0;first=-1;last=-1
        for k in range(offsets[i],offsets[i+1]):
            if codes[k]==65535:continue
            total+=float(distance[k]);used+=1
            if first<0:first=k
            last=k
        if first<0 or total<=0:continue
        den=max(total,1.);r=np.zeros(90,np.float64)
        r[0]=total/1000;r[2]=used/100;r[3]=total/max(chord[i],1.)
        r[4]=distance[first]/1000;r[5]=distance[last]/1000
        for k in range(offsets[i],offsets[i+1]):
            code=int(codes[k])
            if code==65535:continue
            d=float(distance[k]);w=d/den;speed=length[code]/max(free[code],.001)
            r[1]+=d*free[code]/max(length[code],.001)/600
            r[6]+=w*length[code]/1000;r[7]=max(r[7],d);r[8]+=w*speed/10;r[9]+=w*(orientation[code]%2)
            if code==int(codes[first]):r[10]+=w
            if code==int(codes[last]):r[11]+=w
            r[12+orientation[code]]+=w;r[20+speedbin[code]]+=w
            r[26+int((np.int64(code)*2654435761)%64)]+=w
        out[i]=r
    return out
