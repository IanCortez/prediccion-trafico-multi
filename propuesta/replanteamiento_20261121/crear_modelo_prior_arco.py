"""Information-preserving correction: keep time-sampledfeatures, align only the geometryprior."""
from base_replanteo import *
import ast
source=OUT/'atencion_subtrayectos_s0/fuentes/modelo_atencion_subtrayectos.py';s=source.read_text()
s=s.replace('class SubtrajectoryAttention','class ArcPriorAttention')
s=s.replace('def branch_output(self,hidden,raw):','def branch_output(self,hidden,raw,arc_geometry):')
s=s.replace("  R=raw.float();B,K,W=hidden.shape","  R=raw.float();B,K,W=hidden.shape;G=arc_geometry.float()\n  assert G.shape==(len(R),64,14)")
s=s.replace("  R=R.gather(1,order[:,:,None].expand(-1,-1,34))",
 "  R=R.gather(1,order[:,:,None].expand(-1,-1,34));G=G.gather(1,order[:,:,None].expand(-1,-1,14))")
s=s.replace("  geo_loc=pooled[...,:14]+3*torch.tanh(raw_head[...,3:17]/3)",
 "  geometry_prior=torch.einsum('bkrs,bsc->bkrc',weight,G)\n  geo_loc=geometry_prior+3*torch.tanh(raw_head[...,3:17]/3)")
s=s.replace('def forward(self,x,public,history,old_individual,individual):','def forward(self,x,public,history,old_individual,individual,arc_geometry):')
s=s.replace("self.branch_output(hidden[0],individual)","self.branch_output(hidden[0],individual,arc_geometry)")
s=s.replace("self.branch_output(hidden[0][i:i+256],individual[i:i+256])","self.branch_output(hidden[0][i:i+256],individual[i:i+256],arc_geometry[i:i+256])")
s=s.replace('"""Frozen protected30componentparent plus8slots overreal orderedhistoricalsubtrajectories;34Dincludesstart/stopcontext."""',
 '"""Keepall34historicaltime/contextfeatures andsameparameters; use metricarclength geometry asauxiliarycomponentprior."""')
assert s.count('geometry_prior=torch.einsum')==1 and 'arc_geometry[i:i+256]' in s
path=OUT/'modelo_prior_arco_subtrayectos.py';assert not path.exists();ast.parse(s);path.write_text(s)
oldplan=read(OUT/'plan_correccion_arco_subtrayectos.json')
write(OUT/'plan_prior_geometrico_arco.json',dict(
 status='Preparedbefore newgeometry-correction outcome; noGPUtraining launched.',
 reason='Train-onlysemanticproof found212.32mmedian geometryprior mismatch. Correcttheprior withoutdiscarding observedhistoricaltime-profile features.',
 original_inputs='Keepalloriginal34Dtime-sampledsourcefeatures, sourcepathproxy, originalnormalizer43b431... and timeanchor exactly.',
 correction='Supplyseparate14Darclength sourcegeometry onlyto geometryprior. No newneuralparameters. No influenceonTTforward forfixedweights, onlyonjointauxiliarytraininggradients.',
 paired='Samefrozen30componentparent, sameinitialneuralparameters andbuffers, exact6epochs/seed/optimizer/batches. Comparewithtime6control129.990034 andcurrentverifiedbest.',
 source_indices='Same2048nearestpointsets/64parents andalloriginalTrainqueries; no neighbor/bandwidth/label/filter changes.',
 instruments=['OriginalTrainarclengthteacher512parents','892actualpartialintervals','Stationarypoint insertiongeometryinvariance',
 'FixedNNweightsTTbranch independencefromgeometryprior','Pairedinitialparametersexact','Raw65/order/singleton/fullcheckpointreplay'],
 old_prepared_replacement='Arc34replacement trainer preparedbutneverexecuted; superseded before any newscore toretaininformativehistoricaltimefeatures.',
 source_parent='atencion_historia_individual_s0',epochs=6,
 priority='Finishfrozenjoint24trial; preserve/replayqualifiedresult. Then semanticpriorcorrection before spendingonoriginalsourcevaluecontrol. No causalclaim withoutmatchedcontrol.',
 current_joint_result_exists_at_planning=(OUT/'subtrayectos_conjuntos_s0/resultado.json').exists(),
 Test='SEALED',goal='ACTIVE'))
write(OUT/'fuentes_entrenamiento_arco_preparadas.json',dict(prepared_but_not_executed=True,superseded_by='prior_arco_subtrayectos_s0',
 reason='Retainalltime-sampledfeatures; usearclength onlywhere geometryprior mustmatchoriginalteacher. DoNOTrunentrenar_atencion_subtrayectos_arco.py.'))
print('Geometryprior-onlycorrection prepared; old34features andallneuralparameters preserved.')
