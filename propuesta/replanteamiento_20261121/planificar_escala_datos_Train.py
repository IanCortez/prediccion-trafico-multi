"""Prospective data-scale instrument; no additional fitting or heldlabel use."""
from base_replanteo import *
import math
with np.load(OUT/'memoria_puntos_Train_s0/puntos_Train.npz') as z:
 ts=z['timestamp'];time_s=z['time_s'];eligible=z['eligible']
with np.load(OUT/'oraculo_tiempo_calle_train_s0/auxiliares_Train.npz') as z:
 held=z['train_index'][40000:];heldts=z['timestamp'][40000:];oldfit=z['train_index'][:40000]
cutoff=int(heldts.min());fit=np.sort(eligible[ts[eligible]+time_s[eligible]<cutoff])
assert len(np.intersect1d(fit,held))==0 and np.isin(oldfit,fit).all()
updates_per_epoch=math.ceil(len(fit)/256);old_updates=80*math.ceil(40000/256);epochs=math.ceil(old_updates/updates_per_epoch)
plan=dict(status='CONTINGENT_NOT_STARTED',source_fit_n=len(fit),held_n=10000,cutoff=cutoff,all_source_arrivals_before_held=True,
 original_old40000fit_subset_preserved=True,all_original_primary_filters_labels_split_unchanged=True,
 trigger='Current fixed80epoch40k experiment completes and independentreplay confirms failed5sgate with >20s train/heldgeneralization gap; do not rerun solely for a timeout.',
 motivation='Observed no-source40k model generalizes at150.53s despite much smaller trainingerror; this is a data/capacity regime concern, not proof of sourcekernel incapacity.',
 fair_controls='Fresh no-source, geometryproxy andobservedtime models fit identical fullprecutoffTrainrows, same matchedneuralinitialization and original held10k. No pretrainedheldlabels.',
 epoch_budget=dict(old_total_updates=old_updates,new_updates_per_epoch=updates_per_epoch,new_fixed_full_epochs=epochs,
 new_total_updates=epochs*updates_per_epoch,rule='Ceiling of oldtotalupdates divided by fullpassupdates; derived from metadata, not chosen by heldMAE.'),
 numerical_solver='Signedlogtail stablemedian and stablebinNLL alreadyindependentlyverified; preserve original40k results and evaluate its unchangedweights with stablemedian first.',
 new_data='All existing eligibleprimaryTrainbeforeheldcutoff. No newGPSsourcebank and no outsidecohort auxiliary rows.',
 sources='Same original pointbank; wholeparentweekfold excluded BEFORE tree for each fitrow, allsourcearrivalsbeforecutoff.',
 resource_limits=dict(GPU_fraction=.5,batch=256,sourceindices_cache_MiB=350,persistent_rawtokens=False,
 execution='Stream arctokens from existing pointbank/prefix on GPU; do not materialize two all-data rawtoken arrays.'),
 gate='Same>=5s versus matchedfreshbaseline andstoredCatBoostreference,>=2s versus geometryproxy, all exploratorydayCIupper<0; no posthoc threshold reduction.',
 Test='SEALED',Validation=False,no_predictive_result_claim=True)
write(OUT/'plan_escala_datos_Train.json',plan)
print(json.dumps(dict(fit_n=len(fit),held_n=10000,fixed_full_epochs=epochs,old_updates=old_updates,new_updates=epochs*updates_per_epoch,started=False)))
