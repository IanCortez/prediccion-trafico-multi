"""Hierarchical historical competing-risk probabilities. No query duration is used."""
import numpy as np

def event_groups(cell,heading,destination,timestamp,elapsed,age):
    entry=np.asarray(timestamp,dtype=np.int64)+15*(np.asarray(elapsed,dtype=np.int64)-np.asarray(age,dtype=np.int64))
    weekday=(entry//86400+3)%7
    clock=(entry%86400)//14400+6*(weekday>=5)
    return ((np.asarray(cell)*5+np.asarray(heading))*2+(np.asarray(cell)==np.asarray(destination)))*12+clock

def survival(hist):
    return np.cumsum(hist.sum(-1,dtype=np.uint32)[...,::-1],axis=-1,dtype=np.uint32)[...,::-1].copy()

def empirical(hist,risk,prior,strength):
    counts=np.stack([risk.astype(np.float64)-hist.sum(-1,dtype=np.float64),hist[...,0],hist[...,1]],-1)
    assert np.all(counts>=0)
    return (counts+strength*prior)/(risk[...,None]+strength)

class TemporalEventHistory:
    def __init__(self,fold_hist):
        assert fold_hist.ndim==5 and fold_hist.shape[0]==5 and fold_hist.shape[-2:]==(120,2)
        self.fold=fold_hist
        self.groups=fold_hist.shape[1]
        assert self.groups%120==0
        self.N=self.groups//120
        self.total=fold_hist.sum(0,dtype=np.uint32)
        self.risk=survival(self.total)
        self.fold_risk=survival(fold_hist)
        self.base_fold=fold_hist.reshape(5,self.N*10,12,120,2).sum(2,dtype=np.uint32)
        self.base=self.base_fold.sum(0,dtype=np.uint32)
        self.base_risk=survival(self.base)
        self.base_fold_risk=survival(self.base_fold)
        self.global_fold=self.base_fold.reshape(5,self.N,5,2,120,2).sum((1,2),dtype=np.uint32)
        self.global_total=self.global_fold.sum(0,dtype=np.uint32)
        self.global_risk=survival(self.global_total)
        self.global_fold_risk=survival(self.global_fold)
        self.simple=np.array([[.5,.5,0],[.49,.49,.02]])

    def lookup(self,group,age,fold=None):
        group=np.asarray(group,dtype=np.int64)
        age=np.broadcast_to(np.asarray(age,dtype=np.int64),group.shape)
        base=group//12
        flag=base%2
        gh=self.global_total[flag,age].copy()
        gr=self.global_risk[flag,age].copy()
        bh=self.base[base,age].copy()
        br=self.base_risk[base,age].copy()
        lh=self.total[group,age].copy()
        lr=self.risk[group,age].copy()
        if fold is not None:
            fold=np.broadcast_to(np.asarray(fold,dtype=np.int64),group.shape)
            gh-=self.global_fold[fold,flag,age]
            gr-=self.global_fold_risk[fold,flag,age]
            bh-=self.base_fold[fold,base,age]
            br-=self.base_fold_risk[fold,base,age]
            lh-=self.fold[fold,group,age]
            lr-=self.fold_risk[fold,group,age]
        gp=empirical(gh,gr,self.simple[flag],1)
        bp=empirical(bh,br,gp,50)
        return empirical(lh,lr,bp,20).astype(np.float32)

    def complete_probability(self):
        probability=np.empty((self.groups,120,3),np.float32)
        for first in range(0,self.groups,4096):
            last=min(first+4096,self.groups)
            probability[first:last]=self.lookup(np.arange(first,last)[:,None],np.arange(120)[None,:])
        return probability
