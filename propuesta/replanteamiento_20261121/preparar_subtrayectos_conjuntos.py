"""Prepare a separate fully joint fresh-model experiment, not an extension of a frozen branch."""
from base_replanteo import *
p=OUT/'modelo_subtrayectos_conjuntos.py';s=(OUT/'modelo_atencion_subtrayectos.py').read_text()
s=s.replace('class SubtrajectoryAttention','class JointSubtrajectoryAttention')
s=s.replace('  for p in parent.parameters():p.requires_grad_(False)\n','')
s=s.replace('  super().train(mode);self.parent.eval();return self','  super().train(mode);return self')
s=s.replace('def forward(self,x,public,history,old_individual,individual):','def forward(self,x,public,history,individual):')
s=s.replace('with torch.no_grad():base=self.parent(x,public,history,old_individual)','base=self.parent(x,public,history)')
s=s.replace("nn.init.constant_(self.gate[-1].bias,-4.)","nn.init.constant_(self.gate[-1].bias,0.)")
assert 'requires_grad_(False)' not in s and 'with torch.no_grad():base=' not in s
assert not p.exists();p.write_text(s)
plan=dict(architecture='Freshglobal8 +publicroute8 +historicalwholetripprototype6 +orderedsubtrajectoryattention8 =30components',
 difference='All neuralweights initializedfresh and jointlytrained; no frozenin-sampleteacher controlsnewsourcegradient.',
 reason='Protectedparent saw allTrain, so frozen-branchtraining canunderuse crossfit sourceinformation. Treat as optimizationhypothesis, not provenerrorcause.',
 gate_initial_logit=0,epochs=24,lr=.001,scheduler='FixedCosineAnnealingLR T_max24 eta_min1e-5',
 selection='Fixedfinalepoch24; no Validationepochselection. Controlonlyif>=.5s improvement and upperdayCI<0.',
 inputs='Same216publicODfeatures and samehistorical/projectionbanks; raw5 only, noactualquerypath/N/T.',
 primary_train=1023485,Validation=127941,Test='SEALED',
 source_data='Same64orderedsubtrip sources and34Dtokens withwholeparentweekfold exclusion; no neighborcount/radius/bandwidth tuning.',
 modelweights='Fullsmallmodel once; checkpointlatest forrecovery, finalretained<131.06 orstrictbest. Allcompletedmetrics/predictions/config retained.',
 resource='OneheavyGPU .5cap, actualfullresidentdata B512gradientpreflight required.')
write(OUT/'plan_subtrayectos_conjuntos.json',plan)
s=(OUT/'entrenar_atencion_subtrayectos.py').read_text()
s=s.replace('from inferencia_atencion_historia import load_frozen_attention,raw_inputs',
 'from inferencia_historica_fresca import raw_inputs\nfrom modelo_mezcla_rutas import make_duration_model\nfrom modelo_mezcla_proyeccion import ProjectionMixture\nfrom modelo_mezcla_historica import HistoricalRouteMixture\nfrom historicos_rutas import HistoryRouteBank')
s=s.replace("parent,ck,reference,parent_run,proto,oldlibrary,helper,oldtoken=load_frozen_attention('atencion_historia_individual_s0',device='cpu')",
 """previous=read(OUT/'atencion_subtrayectos_s0/resultado.json');assert previous['complete'] and not previous['meaningful_gate_passed']
reference=read(OUT/'nuevo_mejor_verificado.json');parent_run=OUT/reference['id']
base_ctor=torch.load(OUT/'historica_fresca_s0/constructor_base.pt',map_location='cpu',weights_only=False)
assert 'state' not in base_ctor and base_ctor['Train_n']==1023485
hctor=dict(np.load(OUT/'historica_fresca_s0/constructor_historia.npz'))
mctor=dict(np.load(OUT/'historica_fresca_s0/constructor_mapa.npz'))
base_record=read(BEST/'resultado.json')
ck=dict(features=base_ctor['features'],base_config=base_record['config'])
duration=make_duration_model(base_ctor,base_record['config'])
projection=ProjectionMixture(duration,mctor['mean'],mctor['std'],'expanded')
parent=HistoricalRouteMixture(projection,hctor['mean'],hctor['std'],hctor['time_anchor'],'historical')
with torch.no_grad():parent.gate[-1].bias.zero_()
proto=HistoryRouteBank(OUT/'datos_prototipos_historicos_s0/memoria_Train.npz')""")
s=s.replace('from modelo_atencion_subtrayectos import SubtrajectoryAttention','from modelo_subtrayectos_conjuntos import JointSubtrajectoryAttention')
s=s.replace("run=OUT/'atencion_subtrayectos_s0'","run=OUT/'subtrayectos_conjuntos_s0'")
s=s.replace("'entrenar_atencion_subtrayectos.py','modelo_atencion_subtrayectos.py'","'entrenar_subtrayectos_conjuntos.py','modelo_subtrayectos_conjuntos.py'")
s=s.replace("'inferencia_atencion_historia.py'","'inferencia_historica_fresca.py','modelo_mezcla_proyeccion.py','modelo_mezcla_historica.py','historicos_rutas.py','piloto_prototipos_ruta.py'")
s=s.replace('epochs=6,','epochs=24,').replace("plan=read(OUT/'plan_atencion_subtrayectos.json')","plan=read(OUT/'plan_subtrayectos_conjuntos.json')")
s=s.replace("selection='Fixedfinalepoch6 only, no Validationstopping or epochselection'","selection='Allfreshneuralweights, fixedfinalepoch24, no Validationstopping or epochselection',previous_result_sha256=sha(OUT/'atencion_subtrayectos_s0/resultado.json')")
s=s.replace("model=SubtrajectoryAttention(","model=JointSubtrajectoryAttention(")
s=s.replace("public=[];histories=[];oldqueries=[];subqueries=[];indices=[]","public=[];histories=[];subqueries=[];indices=[]")
s=s.replace(" oq=dict(np.load(OUT/'datos_historia_individual_s0'/(split+'_indices.npz')))\n","")
s=s.replace("for z in [cb,hb,oq,sq]","for z in [cb,hb,sq]")
s=s.replace(" oldqueries.append({key:torch.as_tensor(oq[key],device='cuda') for key in ['xy_km','hour','weekend','indices']})\n","")
s=s.replace("del cb,hb,oq,sq","del cb,hb,sq")
s=s.replace("q,vq=oldqueries;","")
s=s.replace("def tokens(query,subquery,indexbank,index):\n old=oldlibrary.features(query['xy_km'][index],query['hour'][index],query['weekend'][index],query['indices'][index])",
 "def tokens(subquery,indexbank,index):")
s=s.replace(" return old,sub"," return sub")
s=s.replace("old,sub=tokens(q,s,si,index);out=model(x[index],c[index],h[index],old,sub)","sub=tokens(s,si,index);out=model(x[index],c[index],h[index],sub)")
s=s.replace("assert all(p.grad is None for p in model.parent.parameters())",
 "assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)\n assert model.duration.backbone.output.weight.grad is not None if hasattr(model.duration.backbone.output,'weight') else any(p.grad is not None and p.grad.norm()>0 for p in model.duration.parameters())")
s=s.replace("model.branch_parameters()","model.parameters()")
s=s.replace("parent_gradients_absent=True","fresh_parent_joint_gradients_finite=True")
s=s.replace("old,sub=tokens(vq,vs,vsi,index)","sub=tokens(vs,vsi,index)")
s=s.replace("out=model.parent(vx[index],vc[index],vh[index],old) if parent_only else model(vx[index],vc[index],vh[index],old,sub)",
 "out=model(vx[index],vc[index],vh[index],sub)")
s=s.replace("del old,sub,out","del sub,out")
s=s.replace("protected=predict(True);assert np.array_equal(protected[:,1],saved['pred'])\nwrite(run/'verificacion_referencia.json',dict(passed=True,all127941parent_points_exact=True,Test=False))",
 "write(run/'verificacion_inicializacion.json',dict(passed=True,no_retained_network_state_loaded=True,base_constructor_sha256=sha(OUT/'historica_fresca_s0/constructor_base.pt'),protected_reference_unchanged=sha(reference['model_file'])==reference['model_sha256'],Test=False))")
s=s.replace("history=[];first_epoch=1","history=[];first_epoch=1\nscheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=24,eta_min=1e-5)")
s=s.replace("check=model.load_state_dict(recovery['branch_state'],strict=False)\n assert not check.unexpected_keys and all(k.startswith('parent.') for k in check.missing_keys)",
 "model.load_state_dict(recovery['state'],strict=True)\n scheduler.load_state_dict(recovery['scheduler'])")
s=s.replace("range(first_epoch,7)","range(first_epoch,25)")
s=s.replace("history.append(dict(epoch=epoch,primary_train_n=count,train_loss=total/count,seconds=time.time()-tick))",
 "scheduler.step()\n history.append(dict(epoch=epoch,primary_train_n=count,train_loss=total/count,lr=opt.param_groups[0]['lr'],seconds=time.time()-tick))")
s=s.replace("branch={key:value for key,value in model.state_dict().items() if not key.startswith('parent.')}","branch=model.state_dict()")
s=s.replace("dict(branch_state=branch,optimizer=opt.state_dict(),epoch=epoch","dict(state=branch,optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch")
s=s.replace("path=run/'epoca_6.pt';torch.save(dict(branch_state=branch,constructor=constructor,config=config,features=names,epoch=6),path)",
 "path=run/'epoca_24.pt';torch.save(dict(state=branch,constructor=constructor,base_constructor=base_ctor,base_config=base_record['config'],history_constructor=hctor,map_constructor=mctor,config=config,features=names,epoch=24),path)")
s=s.replace("X,RC,RH,RQ,rnames=raw_inputs(items,ck['base_config'],proto,oldlibrary,helper,oldtoken)","X,RC,RH,rnames=raw_inputs(items,ck['base_config'],proto)")
s=s.replace(" tq={k:torch.as_tensor(v,device='cuda')[padded] for k,v in RQ.items()}\n old=oldlibrary.features(tq['xy_km'],tq['hour'],tq['weekend'],tq['indices'])\n","")
s=s.replace("out=model(*arrays,old,raw[padded])","out=model(*arrays,raw[padded])")
s=s.replace("_,cached=tokens(vq,vs,vsi,torch.as_tensor(ii,device='cuda'))","cached=tokens(vs,vsi,torch.as_tensor(ii,device='cuda'))")
s=s.replace("fixed_epoch=6","fixed_epoch=24")
s=s.replace("parent_source=reference['id']","comparison_reference=reference['id'],trained_from_scratch=True")
assert 'oldlibrary' not in s and 'oldqueries' not in s and 'load_frozen_attention' not in s and 'branch_state' not in s
p=OUT/'entrenar_subtrayectos_conjuntos.py';assert not p.exists();p.write_text(s)
import ast
for name in ['modelo_subtrayectos_conjuntos.py','entrenar_subtrayectos_conjuntos.py']:ast.parse((OUT/name).read_text(),filename=name)
print('Separatejointfresh-model sources prepared, not launched.')
