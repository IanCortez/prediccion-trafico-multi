from base_replanteo import *
s=(OUT/'entrenar_direccional.py').read_text(encoding='utf-8-sig');s=s.replace('from modelo_direccional import DirectionalArrival,heading_ids','from modelo_dir_prior import PriorDirectionalArrival\nfrom modelo_direccional import heading_ids');s=s.replace("model=DirectionalArrival(stats,grid,graph,distance,stats['anchors'][3]).cuda()","prior_counts=np.load(OUT/'prior_transiciones/conteos_fold.npz')['counts'];model=PriorDirectionalArrival(stats,grid,graph,distance,stats['anchors'][3],prior_counts).cuda();del prior_counts")
s=s.replace("'entrenar_direccional.py','modelo_direccional.py'","'entrenar_dir_prior.py','modelo_dir_prior.py','modelo_direccional.py'")
s=s.replace("model.state_parameters(q,x[ix,:4],safe,dd,heading)","model.state_parameters(q,x[ix,:4],safe,dd,heading,trainfold[ix])")
s=s.replace("origin=torch.tensor(ids(tr,'o'),device='cuda');","trainfold=torch.tensor(((tr['timestamp']//(7*86400))%5).astype(np.int64),device='cuda');origin=torch.tensor(ids(tr,'o'),device='cuda');")
s=s.replace("model.posterior(x[ix],origin[ix],dest[ix])","model.posterior(x[ix],origin[ix],dest[ix],excluded_fold=trainfold[ix])")
s=s.replace("dp_batch=2,inference_batch=16","dp_batch=2,inference_batch=8").replace('range(0,len(use),16)','range(0,len(use),8)').replace('use[first:first+16]','use[first:first+8]').replace('if n<16:','if n<8:').replace('expand(16-n)','expand(8-n)')
s=s.replace("local_loss='next-zone CE","transition_prior='Hierarchical Train counts over cell/destination bearing-distance/incoming heading; subtract entire departure-week fold at all Train local and total-loss steps; Validation uses all Train; count exclusion does not exclude neural fitting or allowed graph',local_loss='next-zone CE")
s=s.replace("sha(OUT/'modelo_direccional.py')","sha(OUT/'modelo_dir_prior.py')").replace("data_grid_sha256=sha(OUT/'datos_semi_markov/resultado.json'),","data_grid_sha256=sha(OUT/'datos_semi_markov/resultado.json'),prior_counts_sha256=sha(OUT/'prior_transiciones/conteos_fold.npz'),")
assert 'PriorDirectionalArrival' in s and 'excluded_fold=trainfold[ix]' in s
(OUT/'entrenar_dir_prior.py').write_text(s,encoding='utf-8');print('Separate empirical-prior trainer prepared.')

