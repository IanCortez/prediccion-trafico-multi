from common import *
import os
os.environ["HF_HOME"]=str(ROOT/".runtime/huggingface")
os.environ["HF_HUB_CACHE"]=str(ROOT/".runtime/huggingface/hub")
os.environ["TRANSFORMERS_CACHE"]=str(ROOT/".runtime/huggingface/transformers")
os.environ["TOKENIZERS_PARALLELISM"]="false"
from huggingface_hub import HfApi,snapshot_download
from transformers import AutoTokenizer,AutoModel
import torch
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
started=time.time();repo="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
revision=HfApi().model_info(repo).sha
folder=BASE/"modelos/minilm_multilingue";folder.mkdir(parents=True,exist_ok=True)
snapshot_download(repo_id=repo,revision=revision,local_dir=str(folder),cache_dir=os.environ["HF_HUB_CACHE"],allow_patterns=["*.json","*.txt","*.safetensors","sentencepiece.bpe.model","tokenizer.*"])
texts=json.loads((DATA/"textos_urbanos.json").read_text(encoding="utf-8-sig"))
tokenizer=AutoTokenizer.from_pretrained(str(folder),local_files_only=True)
model=AutoModel.from_pretrained(str(folder),local_files_only=True,use_safetensors=True).cuda().half().eval()
emb=[]
with torch.no_grad():
    for start in range(0,len(texts),128):
        token=tokenizer(texts[start:start+128],padding=True,truncation=True,max_length=96,return_tensors="pt")
        token={k:v.cuda() for k,v in token.items()}
        h=model(**token).last_hidden_state.float();mask=token["attention_mask"][...,None]
        pooled=(h*mask).sum(1)/mask.sum(1).clamp_min(1);pooled=torch.nn.functional.normalize(pooled,dim=1)
        emb.append(pooled.cpu().numpy())
emb=np.concatenate(emb)
pca=PCA(n_components=64,whiten=True,random_state=0);projected=pca.fit_transform(emb).astype(np.float32)
np.savez_compressed(DATA/"embeddings_textuales.npz",embedding=emb,reduced=projected,pca_components=pca.components_,pca_mean=pca.mean_,pca_variance=pca.explained_variance_)
del model;torch.cuda.empty_cache()
base=np.load(DATA/"contexto_rejilla25m.npz");info={k:int(base[k]) for k in ("gridstep","x0","y0","nx","ny")}
gx,gy=np.meshgrid(info["x0"]+(np.arange(info["nx"])+.5)*info["gridstep"],info["y0"]+(np.arange(info["ny"])+.5)*info["gridstep"],indexing="ij")
grid=np.column_stack([gx.ravel(),gy.ravel()])
z=np.load(DATA/"entidades_textuales.npz");parts=[base["features"]];names=base["names"].tolist()
for kind,label in ((0,"poi_text"),(1,"road_text")):
    subset=np.flatnonzero(z["kind"]==kind);tree=cKDTree(z["xy"][subset])
    neighbors=tree.query_ball_point(grid,100,workers=4);counts=np.array([len(v) for v in neighbors])
    ptr=np.r_[0,np.cumsum(counts)];col=np.concatenate(neighbors).astype(np.int32)
    weights=z["weight"][subset[col]]
    mat=csr_matrix((weights,col,ptr),shape=(len(grid),len(subset)))
    denom=np.asarray(mat.sum(1)).ravel()
    mean=(mat@projected[z["text_idx"][subset]])/np.maximum(denom[:,None],1)
    parts.extend([mean.astype(np.float32),(counts>0).astype(np.float32)[:,None],np.log1p(counts).astype(np.float32)[:,None]])
    names.extend([f"{label}_pc{i}" for i in range(64)]+[label+"_available",label+"_logcount"])
out=np.column_stack(parts)
np.savez_compressed(DATA/"contexto_semantico_rejilla25m.npz",features=out,names=np.array(names),**info)
write_json(DATA/"contexto_semantico_manifiesto.json",dict(model=repo,revision=revision,model_card="https://huggingface.co/"+repo,embedding_dimensions=384,reduced_dimensions=64,reduction="PCA whitening fit only static text vectors",pooling="100m, POI equal weight, road segment physical-length weight",texts_sha256=sha(DATA/"textos_urbanos.json"),grid_sha256=sha(DATA/"contexto_semantico_rejilla25m.npz"),n_features=out.shape[1],no_trip_labels=True,seconds=time.time()-started))
print(dict(shape=out.shape,seconds=time.time()-started,revision=revision),flush=True)

