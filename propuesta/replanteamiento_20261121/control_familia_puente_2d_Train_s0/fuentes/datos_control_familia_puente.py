"""Samepublicfeatures/fitnormalization, with originalGPSbank forbidden and not needed."""
from base_replanteo import *
import torch,gc
from inferencia import feature_input
class PublicControlInputs:
 def __init__(self,run):
  source=OUT/'puente_temporal_2d_Train_s0'
  metadata=torch.load(source/'constructor.pt',map_location='cpu',weights_only=False)
  file=OUT/'indices_escala_Train_s0/metadatos_Train.npz';assert sha(file)==metadata['fit_metadata_sha256']
  with np.load(file) as z:
   self.index=z['train_index'];self.row=z['row'];self.timestamp=z['timestamp'];self.nfit=int(z['fit_n'])
  assert self.nfit==951718 and len(self.index)==self.nfit+10000
  d,X,names=feature_input('train',read(BEST/'resultado.json'))
  keep=np.array([not name.startswith('mm_') for name in names]);self.names=[n for n,k in zip(names,keep) if k]
  assert self.names==metadata['features'] and len(self.names)==206
  X=np.ascontiguousarray(X[self.index][:,keep]);assert np.isfinite(X).all()
  assert np.array_equal(d['row'][self.index],self.row) and np.array_equal(d['timestamp'][self.index],self.timestamp)
  self.query={k:d[k][self.index] for k in ['o_lon','o_lat','d_lon','d_lat','timestamp']}
  self.truth=np.load(DATA/'train_y.npy')[self.index].astype(np.float32)
  assert np.all(self.timestamp[:self.nfit]+self.truth[:self.nfit]<metadata['source_cutoff'])
  self.constructor=metadata['constructor']
  self.x=torch.tensor(X,device='cuda')
  self.target_n=torch.tensor((self.truth[:self.nfit]/15).astype(np.int64),device='cuda')
  torch.save(dict(constructor=self.constructor,features=self.names,fit_n=self.nfit,held_n=10000,
   shared_fit_metadata_sha256=metadata['fit_metadata_sha256'],shared_constructor_sha256=sha(source/'constructor.pt')),run/'constructor.pt')
  del X,d;gc.collect()
