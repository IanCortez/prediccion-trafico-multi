"""Preregister an orthogonal density-head control before the first currentheldresult."""
from base_replanteo import *
active=OUT/'puente_temporal_2d_Train_s0'
assert active.exists() and not (active/'none/resultado.json').exists(),'Must freeze before any newheldmetric'
path=OUT/'protocolo_control_familia_puente_2d.json';assert not path.exists()
protocol=dict(status='FROZEN_BEFORE_FIRST_BRIDGE_HELDRESULT',
 question='Does the endpointStudentvelocity density head itself help, independently of privatepathsupervision?',
 reason='The main three-arm protocol isolates auxiliarymotion information within a fixednewhead. It cannot byitself attribute gains to the analytichead versus a conventionaldensity family.',
 preserve_main_gate_unchanged=True,main_protocol_sha256=sha(active/'protocolo.json'),
 candidate='The noauxiliary arm of puente_temporal_2d_Train_s0 only; do not replace a failedauxiliarygate with thiscontrast.',
 matched_control='Fresh8lognormalcomponents x4publicencoders, exactsame initialquerybackbone weights, public206fitnormalizers/physicalanchors, seed20261130, epochorders/dropoutseeds, AdamWcosine recipe.',
 objective='Same properordinalCRPS_seconds/600+.02original15secondbinNLL. Lognormal probabilitiesintegrated overcentered15sbins with foldedtails; bridge massdefined on originalfixedn4..119.',
 fit_n=951718,held_n=10000,epochs=4,batch=256,updates_per_arm=14872,
 schedule='One matchedlognormal control after mainthree-arm actualterminalresult and replay. No parallelheavyGPU.',
 gate='NoauxStudent>=5s betterthan the matchedlognormal control AND savedpublicNN130.029068616394s; both exploratorypaired-dayCIupper<0.',
 secondary_gate_does_not_lower_main_auxiliary_gate=True,
 interpretation='A positivecontrast qualifies the entirefixedpublichead recipe, not an isolatedcausal effect of oneequation or a73.5claim.',
 limits=['Oneinitialization, reusedheldTrain; notconfirmation.','Differentdensityhead parametercounts and probabilityfamilies; identicalquerybackbone initialization and commonproper scoringobjective.'],
 no_seed_width_component_epoch_or_lossweight_grid=True,
 no_actualqueryGPS_N_T=True,Test='SEALED',Validation=False,code_sha256=sha(__file__))
write(path,protocol);print(json.dumps(dict(registered_before_any_currentheldresult=True,main_gate_unchanged=True,one_matchedheadcontrol_planned=True)))
