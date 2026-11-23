"""Keep timefeatures; train a separatelyfrozen geometryprior correction with identical neuralcapacity."""
from base_replanteo import *
import ast
s=(OUT/'entrenar_atencion_subtrayectos_arco.py').read_text()
s=s.replace("from tokens_subtrayectos import query_parts_sub","from tokens_subtrayectos import SubtripTokenBank,query_parts_sub")
s=s.replace("from modelo_atencion_subtrayectos import SubtrajectoryAttention",
 "from modelo_atencion_subtrayectos import SubtrajectoryAttention\nfrom modelo_prior_arco_subtrayectos import ArcPriorAttention")
s=s.replace("normal=read(OUT/'constructor_tokens_arco_s0/resultado.json')","normal=read(folder/'normalizacion_resultado.json')")
s=s.replace("OUT/'constructor_tokens_arco_s0/constructor_tokens.npz'","folder/'constructor_tokens.npz'")
s=s.replace("run=OUT/'atencion_subtrayectos_arco_s0'","run=OUT/'prior_arco_subtrayectos_s0'")
s=s.replace("'entrenar_atencion_subtrayectos_arco.py'","'entrenar_prior_arco_subtrayectos.py'")
s=s.replace("'modelo_atencion_subtrayectos.py',","'modelo_atencion_subtrayectos.py','modelo_prior_arco_subtrayectos.py',")
s=s.replace("plan=read(OUT/'plan_correccion_arco_subtrayectos.json')","plan=read(OUT/'plan_prior_geometrico_arco.json')")
s=s.replace("best_reference=read(OUT/'nuevo_mejor_verificado.json');time_reference=read(OUT/'atencion_subtrayectos_s0/resultado.json')",
 """if args.resume:best_reference=read(OUT/read(run/'protocolo.json')['comparison_best_id']/'resultado.json')
else:best_reference=read(OUT/'nuevo_mejor_verificado.json')
time_reference=read(OUT/'atencion_subtrayectos_s0/resultado.json')""")
s=s.replace("geometry_sampling='Originalnormalizedmetricarclength and fullsourceintervallength'",
 "geometry_prior='Independent14Dmetricarclengthsourceprior; original34time/contextfeatures andnormalizer unchanged'")
s=s.replace("model=SubtrajectoryAttention(parent,constructor['mean'],constructor['std'],constructor['time_anchor'],'individual')",
 "model=ArcPriorAttention(parent,constructor['mean'],constructor['std'],constructor['time_anchor'],'individual')")
s=s.replace("all_time_context_normalizers_exact=np.array_equal(constructor['mean'][16:],old_ctor['mean'][16:]) and np.array_equal(constructor['std'][16:],old_ctor['std'][16:])",
 "all_normalizers_exact=np.array_equal(constructor['mean'],old_ctor['mean']) and np.array_equal(constructor['std'],old_ctor['std'])")
s=s.replace("model=model.cuda()",
 """assert read(OUT/'verificacion_prior_arco_CPU.json')['passed']
assert read(OUT/'verificacion_prior_arco_CPU.json')['model_sha256']==sha(OUT/'modelo_prior_arco_subtrayectos.py')
assert read(OUT/'verificacion_tokens_arco_Train.json')['passed']
model=model.cuda()""")
s=s.replace("library=SubtripArcLengthBank();tr,X,names=feature_input('train',read(BEST/'resultado.json'))",
 "library=SubtripTokenBank();arc_library=SubtripArcLengthBank(shared=library);tr,X,names=feature_input('train',read(BEST/'resultado.json'))")
probe=" probe_sraw=library.features(probe_sub['xy_m'],probe_sub['hour'],probe_sub['weekend'],probe_indices['parent'],probe_indices['first'],probe_indices['last'])"
assert probe in s;s=s.replace(probe,probe+"\n probe_araw=arc_library.features(probe_sub['xy_m'],probe_sub['hour'],probe_sub['weekend'],probe_indices['parent'],probe_indices['first'],probe_indices['last'])[:,:,:14]")
s=s.replace("probe_oraw,probe_sraw)","probe_oraw,probe_sraw,probe_araw)")
s=s.replace("probe_sraw,probe_out,probe_cpu","probe_sraw,probe_araw,probe_out,probe_cpu")
s=s.replace(" return old,sub"," arc_geometry=arc_library.features(subquery['xy_m'][index],subquery['hour'][index],subquery['weekend'][index],\n  indexbank['parent'][index],indexbank['first'][index],indexbank['last'][index])[:,:,:14]\n return old,sub,arc_geometry")
s=s.replace("old,sub=tokens(q,s,si,index);out=model(x[index],c[index],h[index],old,sub)",
 "old,sub,arc_geometry=tokens(q,s,si,index);out=model(x[index],c[index],h[index],old,sub,arc_geometry)")
s=s.replace("old,sub=tokens(vq,vs,vsi,index)","old,sub,arc_geometry=tokens(vq,vs,vsi,index)")
s=s.replace("else model(vx[index],vc[index],vh[index],old,sub)","else model(vx[index],vc[index],vh[index],old,sub,arc_geometry)")
s=s.replace("del old,sub,out","del old,sub,arc_geometry,out")
line=" raw=library.features(st['xy_m'],st['hour'],st['weekend'],source[0],source[1],source[2])"
assert line in s;s=s.replace(line,line+"\n arc_raw=arc_library.features(st['xy_m'],st['hour'],st['weekend'],source[0],source[1],source[2])[:,:,:14]")
s=s.replace("out=model(*arrays,old,raw[padded])","out=model(*arrays,old,raw[padded],arc_raw[padded])")
s=s.replace("return pred,raw.cpu().numpy(),source","return pred,raw.cpu().numpy(),source,arc_raw.cpu().numpy()")
s=s.replace("raw,rawtokens,rs=raw_predict(items);reverse,rev_tokens,_=raw_predict(items[::-1]);single,single_tokens,_=raw_predict(items[:1])",
 "raw,rawtokens,rs,rawgeo=raw_predict(items);reverse,rev_tokens,_,revgeo=raw_predict(items[::-1]);single,single_tokens,_,singlegeo=raw_predict(items[:1])")
s=s.replace("_,cached=tokens(vq,vs,vsi,torch.as_tensor(ii,device='cuda'));assert np.array_equal(rawtokens,cached.cpu().numpy())",
 "_,cached,cachedgeo=tokens(vq,vs,vsi,torch.as_tensor(ii,device='cuda'));assert np.array_equal(rawtokens,cached.cpu().numpy())\nassert np.array_equal(rawgeo,cachedgeo.cpu().numpy()) and np.array_equal(rawgeo,revgeo[::-1]) and np.array_equal(rawgeo[:1],singlegeo)")
s=s.replace("raw65_cached_sources_tokens_full_quantiles_exact=True,","raw65_cached_sources_tokens_full_quantiles_exact=True,raw65_arc_geometry_exact=True,all34input_features_normalizer_preserved=True,")
assert 'ArcPriorAttention(parent' in s and s.count('return old,sub,arc_geometry')==1
assert "constructor=dict(np.load(folder/'constructor_tokens.npz'))" in s and 'probe_araw)' in s
assert 'old,sub=tokens(' not in s and 'return pred,raw.cpu().numpy(),source,arc_raw.cpu().numpy()' in s
ast.parse(s)
path=OUT/'entrenar_prior_arco_subtrayectos.py';assert not path.exists();path.write_text(s)
write(OUT/'fuentes_prior_arco_preparadas.json',dict(prepared=True,not_launched=True,
 source_sha256=sha(path),model_sha256=sha(OUT/'modelo_prior_arco_subtrayectos.py'),
 semantic_proof=sha(OUT/'verificacion_tokens_arco_Train.json'),fixed_weight_TT_invariance_proof=sha(OUT/'verificacion_prior_arco_CPU.json'),
 original34features_and_constructor_preserved=True,no_new_neural_parameters=True,
 expected_next='Aftercurrentjoint24terminalresultandqualifiedreload, runthis6epochgeometrypriorcorrection. DoNOTlaunchsupersededArc34replacementtrainer.',
 Test='SEALED'))
print('Informationpreserving sixepoch geometryprior correction frozen; no currentexperiment modified.')
