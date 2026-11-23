"""Compactinteger sourcecounts; fold/hour retained until BEFORE everykernel/backoff construction."""
import numpy as np
def aggregate_keys(keys,weights=None):
 keys=np.asarray(keys,np.uint64)
 if weights is None:
  unique,count=np.unique(keys,return_counts=True)
  assert count.max(initial=0)<=np.iinfo(np.uint32).max
  return unique,count.astype(np.uint32)
 weights=np.asarray(weights,np.uint64);assert weights.shape==keys.shape
 if not len(keys):return keys,np.empty(0,np.uint32)
 order=np.argsort(keys,kind='stable');ordered=keys[order]
 start=np.r_[0,np.flatnonzero(ordered[1:]!=ordered[:-1])+1]
 count=np.add.reduceat(weights[order],start)
 assert count.max(initial=0)<=np.iinfo(np.uint32).max
 return ordered[start],count.astype(np.uint32)
def encode_pairs(context,source,destination,state_count):
 base=np.uint64(state_count)
 return (np.asarray(context,np.uint64)*base+np.asarray(source,np.uint64))*base+np.asarray(destination,np.uint64)
def decode_pairs(keys,state_count):
 base=np.uint64(state_count);keys=np.asarray(keys,np.uint64)
 destination=keys%base;temporary=keys//base
 return (temporary//base).astype(np.int16),(temporary%base).astype(np.int64),destination.astype(np.int64)
def encode_events(context,state,state_count):
 return np.asarray(context,np.uint64)*np.uint64(state_count)+np.asarray(state,np.uint64)
def select_fold(keys,counts,state_count,excluded_fold=5,pairs=True):
 assert 0<=excluded_fold<=5
 context=np.asarray(keys,np.uint64)//(np.uint64(state_count)**np.uint64(2) if pairs else np.uint64(state_count))
 selected=np.ones(len(keys),bool) if excluded_fold==5 else context//np.uint64(168)!=excluded_fold
 return np.asarray(keys)[selected],np.asarray(counts)[selected]
def position_counts(keys,counts,kinetic_state_count):
 context,source,destination=decode_pairs(keys,kinetic_state_count)
 return aggregate_keys(encode_pairs(context,source//3,destination//3,kinetic_state_count//3),counts)
