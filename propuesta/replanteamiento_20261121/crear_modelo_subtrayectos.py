"""Freeze an explicit architecture fork; never edit the protected model."""
from base_replanteo import *
src=OUT/'atencion_historia_individual_s0/fuentes/modelo_atencion_historia.py'
assert sha(src)==read(OUT/'atencion_historia_individual_s0/resultado.json')['sources']['modelo_atencion_historia.py']
s=src.read_text();s=s.replace('class IndividualHistoryAttention','class SubtrajectoryAttention')
s=s.replace('nn.Linear(28,64)','nn.Linear(34,64)').replace('nn.Linear(605,128)','nn.Linear(611,128)').replace('nn.Linear(568,64)','nn.Linear(580,64)')
s=s.replace('range(27,-1,-1)','range(33,-1,-1)').replace('expand(-1,-1,28)','expand(-1,-1,34)')
s=s.replace('def forward(self,x,public,history,individual):','def forward(self,x,public,history,old_individual,individual):')
s=s.replace('shape==(len(x),64,28)','shape==(len(x),64,34)')
s=s.replace("   from historia_individual import compress_source_values\n   individual=compress_source_values(individual,history)",
 "   individual=individual.clone();individual[...,:17]=self.token_mean[:17].to(individual.dtype)")
s=s.replace('base=self.parent(x,public,history)','base=self.parent(x,public,history,old_individual)')
s=s.replace('"""Frozen22component parent plus8query-conditioned slots over64individual source trips."""',
 '"""Frozen protected30componentparent plus8slots overreal orderedhistoricalsubtrajectories;34Dincludesstart/stopcontext."""')
target=OUT/'modelo_atencion_subtrayectos.py'
if target.exists():assert target.read_text()==s
else:target.write_text(s)
write(OUT/'plan_atencion_subtrayectos.json',dict(
 source='Implicitorderedintervals of ALLpastTrainrawtrajectories; never queryactualpath',
 hypothesis='Directsubtrajectorykernel failed163.946s; evidenceblend130.960s. Learn source-to-query time transport, route-shape relevance, andconditionalgate from originalTrainqueries.',
 architecture='Frozenprotected30components +8attentioncomponents over64subtrajectorysources;34Dtokens includes7pointgeometry andstart/stopcontext.',
 Train='1023485originalqueries withoriginalT, wholeparentweekfold excludedbeforetree; no narrowingprimarycohort',
 training=dict(epochs=6,batch=512,lr=.001,seed=20261122,loss='NLL+5RPS+.2conditionalTrainGeometry',selection='Fixedfinalepoch6; no Validationepochselection'),
 normalizer='Fixed32768Trainqueries seed5833, sourcefeaturestatistics and robust logsourceT tologqueryT regression; noValidationlabels.',
 main_control_trigger='>=.5seconds improvement overprotected130.158 and upperdayCI<0; then matched sourcevalues control.',
 Test='SEALED',bounded_storage='Onlyparentint32+first/lastuint8indices;34Dtokens derivedonGPU, no token/CDFcache.'))
