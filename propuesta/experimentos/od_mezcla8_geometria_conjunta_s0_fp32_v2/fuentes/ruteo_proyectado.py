from common import *
from map_matching import node_connections,node_distance
from trazado_vial import collect_node_path,add_arc
from numba import njit

def free_connections(comp):
    outj,outl,inj,inl,chain,pos,_=node_connections(comp);prefix=comp['prefix_free']
    for slot in range(2):
        valid=chain[:,slot]>=0;c=chain[valid,slot];p=pos[valid,slot];outl[valid,slot]=comp['chain_free'][c]-prefix[p];inl[valid,slot]=prefix[p]
    return outj,outl,inj,inl,chain,pos,prefix

@njit(cache=True)
def collect_cost_path(ea,fa,eb,fb,u,v,length,arc_cost,D,conn,routing,ids,parts):
    outj,outl,inj,inl,chain,pos,prefix=conn;remain_cost=(1-fa)*arc_cost[ea];tail_cost=fb*arc_cost[eb];a=v[ea];b=u[eb];first=(1-fa)*length[ea];last=fb*length[eb];which=0
    best=remain_cost+node_distance(a,b,D,outj,outl,inj,inl,chain,pos,prefix)+tail_cost
    if ea==eb and fb>=fa:
        cost=(fb-fa)*arc_cost[ea]
        if cost<best:best=cost;which=4
    if fa<=1e-7:
        cost=node_distance(u[ea],u[eb],D,outj,outl,inj,inl,chain,pos,prefix)+tail_cost
        if cost<best:best=cost;which=1;a=u[ea];b=u[eb];first=0.;last=fb*length[eb]
    if fb>=1-1e-7:
        cost=remain_cost+node_distance(v[ea],v[eb],D,outj,outl,inj,inl,chain,pos,prefix)
        if cost<best:best=cost;which=2;a=v[ea];b=v[eb];first=(1-fa)*length[ea];last=0.
    if fa<=1e-7 and fb>=1-1e-7:
        cost=node_distance(u[ea],v[eb],D,outj,outl,inj,inl,chain,pos,prefix)
        if cost<best:best=cost;which=3;a=u[ea];b=v[eb];first=0.;last=0.
    if not np.isfinite(best):return -1,best
    if which==4:return add_arc(ea,(fb-fa)*length[ea],ids,parts,0),best
    used=add_arc(ea,first,ids,parts,0)
    if used<0:return -1,best
    used=collect_node_path(a,b,D,conn,routing,length,ids,parts,used)
    if used<0:return -1,best
    return add_arc(eb,last,ids,parts,used),best

@njit(cache=True)
def path_shape(origin,destination,projected_origin,ids,parts,used,xy,u,v,length,vertices,cumulative,out):
    vertices[0]=origin;vertices[1]=projected_origin;current=projected_origin.copy()
    for k in range(used):
        edge=ids[k];current=current+(xy[v[edge]]-xy[u[edge]])*(parts[k]/length[edge]);vertices[k+2]=current
    m=used+3;vertices[m-1]=destination;cumulative[0]=0.;angles=0.;signed=0.;turns=0.;maximum=0.;lastx=0.;lasty=0.
    for j in range(1,m):
        vx=vertices[j,0]-vertices[j-1,0];vy=vertices[j,1]-vertices[j-1,1];L=np.sqrt(vx*vx+vy*vy);cumulative[j]=cumulative[j-1]+L
        if L>1e-6:
            if lastx*lastx+lasty*lasty>1e-12:
                angle=np.arctan2(lastx*vy-lasty*vx,lastx*vx+lasty*vy);angles+=abs(angle);signed+=angle;turns+=abs(angle)>np.pi/4;maximum=max(maximum,abs(angle))
            lastx=vx;lasty=vy
    total=cumulative[m-1];cursor=1
    for j in range(1,8):
        q=j/8.;target=total*q
        while cursor<m-1 and cumulative[cursor]<target:cursor+=1
        alpha=(target-cumulative[cursor-1])/max(cumulative[cursor]-cumulative[cursor-1],1e-9);point=vertices[cursor-1]*(1-alpha)+vertices[cursor]*alpha;chord=origin*(1-q)+destination*q;out[(j-1)*2:j*2]=(point-chord)/1000
    od=np.sqrt(((destination-origin)**2).sum());out[14]=total/1000;out[15]=np.log(max(total/max(od,1),1));out[16]=angles/np.pi;out[17]=signed/np.pi;out[18]=turns;out[19]=maximum/np.pi;out[20]=1.
    return total
