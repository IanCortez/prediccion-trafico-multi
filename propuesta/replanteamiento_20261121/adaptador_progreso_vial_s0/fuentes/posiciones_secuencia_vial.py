"""Recover physical chain entry/exit and mask unsupported macroactions; never modify time labels."""
import numpy as np
from numba import njit
@njit(cache=True)
def raw_node_at(chain,position,offsets,prefix,nodes,tolerance):
 a=offsets[chain];b=offsets[chain+1];p=np.searchsorted(prefix[a:b],position)+a
 best=-1;error=1e100
 for j in range(max(a,p-1),min(b,p+2)):
  delta=abs(prefix[j]-position)
  if delta<error:error=delta;best=j
 if error>tolerance:return -1
 return nodes[best]
@njit(cache=True)
def chain_position_at_node(chain,node,prefer_end,offsets,prefix,nodes):
 a=offsets[chain];b=offsets[chain+1]
 if prefer_end and nodes[b-1]==node:return prefix[b-1]
 if nodes[a]==node:return prefix[a]
 for j in range(a+1,b):
  if nodes[j]==node:return prefix[j]
 return np.nan
@njit(cache=True)
def assign_position(i,entry,exit,known,code,distance,length,complex_token,candidate_entry,candidate_exit,tolerance):
 c=code[i]
 if c==65535 or complex_token[i] or not np.isfinite(candidate_entry) or not np.isfinite(candidate_exit):return 0
 if candidate_entry< -tolerance or candidate_exit>length[c]+tolerance or candidate_exit< candidate_entry-tolerance:return 0
 e=min(max(candidate_entry,0.),length[c]);x=min(max(candidate_exit,0.),length[c])
 if abs((x-e)-distance[i])>tolerance:return 0
 if known[i]:
  if abs(entry[i]-e)>tolerance or abs(exit[i]-x)>tolerance:return -1
  return 0
 entry[i]=e;exit[i]=x;known[i]=True;return 1
@njit(cache=True)
def recover_positions(code,distance,trip_offsets,initial_chain,initial_pos,terminal_chain,terminal_pos,
 length,src,dst,chain_offsets,prefix,nodes):
 n=len(code);entry=np.zeros(n,np.float64);exit=np.zeros(n,np.float64);known=np.zeros(n,np.bool_);complex_token=np.zeros(n,np.bool_)
 macro=np.zeros(n,np.bool_);reset=np.ones(n,np.bool_);eos=np.zeros(len(trip_offsets)-1,np.bool_);conflicts=0;tolerance=.1
 for i in range(n):
  c=code[i]
  if c==65535:continue
  if distance[i]>length[c]+tolerance:complex_token[i]=True;continue
  if src[c]!=dst[c] and abs(distance[i]-length[c])<=.01:
   entry[i]=0.;exit[i]=length[c];known[i]=True
 for trip in range(len(eos)):
  a=trip_offsets[trip];b=trip_offsets[trip+1];last=b-1
  if code[a]!=65535 and code[a]==initial_chain[trip]:
   c=code[a];e=float(initial_pos[trip]);x=e+distance[a]
   if src[c]==dst[c] and x>length[c]+tolerance:complex_token[a]=True;known[a]=False
   else:
    ans=assign_position(a,entry,exit,known,code,distance,length,complex_token,e,x,tolerance)
    if ans<0:conflicts+=1
  terminal=float(terminal_pos[trip])
  if code[last]!=65535 and code[last]==terminal_chain[trip]:
   c=code[last]
   # A closedchain's endpoint0 andL are the same publicphysicalnode.
   if src[c]==dst[c] and terminal<=1e-4:terminal=length[c]
   e=terminal-distance[last]
   if src[c]==dst[c] and e< -tolerance:complex_token[last]=True;known[last]=False
   else:
    ans=assign_position(last,entry,exit,known,code,distance,length,complex_token,e,terminal,tolerance)
    if ans<0:conflicts+=1
  # Full nonloop tokens and observedfirst/last positions are anchors; no actualduration is read.
  for sweep in range(2):
   for i in range(a,b-1):
    if not known[i] or code[i+1]==65535:continue
    node=raw_node_at(code[i],exit[i],chain_offsets,prefix,nodes,tolerance)
    if node<0:continue
    e=chain_position_at_node(code[i+1],node,False,chain_offsets,prefix,nodes)
    ans=assign_position(i+1,entry,exit,known,code,distance,length,complex_token,e,e+distance[i+1],tolerance)
    if ans<0:conflicts+=1
   for i in range(b-1,a,-1):
    if not known[i] or code[i-1]==65535:continue
    node=raw_node_at(code[i],entry[i],chain_offsets,prefix,nodes,tolerance)
    if node<0:continue
    x=chain_position_at_node(code[i-1],node,True,chain_offsets,prefix,nodes)
    ans=assign_position(i-1,entry,exit,known,code,distance,length,complex_token,x-distance[i-1],x,tolerance)
    if ans<0:conflicts+=1
  for i in range(a,b-1):
   if not known[i] or not known[i+1]:continue
   macro[i]=abs(exit[i]-length[code[i]])<=tolerance and abs(entry[i+1])<=tolerance and dst[code[i]]==src[code[i+1]]
   reset[i+1]=not macro[i]
  if known[last] and code[last]==terminal_chain[trip]:
   eos[trip]=abs(exit[last]-terminal)<=tolerance and entry[last]<=terminal+tolerance
 return entry,exit,known,macro,reset,eos,complex_token,conflicts
