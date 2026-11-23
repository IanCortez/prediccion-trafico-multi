"""Post-fixedresult diagnostic only; cannot change gates, protocols, filters or selection."""
from base_replanteo import *
run=OUT/'red_cinetica_Train_s0';r=read(run/'resultado.json');assert r['complete']
saved=dict(np.load(run/'predicciones_held_Train.npz'))
inputs=OUT/'indices_red_cinetica_Train_s0/consultas_predicciones.npz';assert sha(inputs)==r['input_sha256']
predictions={};summary={}
with np.load(inputs) as z:
 nfit=int(z['fit_n']);p0=z['reference_probability'][nfit:]
 for kind in ['position','kinetic']:
  probability=z[kind+'_probability'][nfit:].astype(np.float64);supported=z[kind+'_supported'][nfit:];mass=z[kind+'_support'][nfit:]
  probability=np.divide(probability,probability.sum(1)[:,None],out=np.zeros_like(probability),where=probability.sum(1)[:,None]>0)
  probability[~supported]=p0[~supported]
  cdf=probability.cumsum(1);pred=(np.argmax(cdf>=.5,axis=1)+4)*15
  predictions[kind+'_graph']=pred
  gated=dict(np.load(run/kind/'predicciones_held_Train.npz'));gate=gated['gate']
  summary[kind]=dict(rawgraph_with_reference_fallback_metrics=metric(saved['true'],pred),
   gated_metrics=r['metrics'][kind],all10000_rows_preserved=True,unsupported_fallback_count=int((~supported).sum()),
   graph_prediction_bias_s=float(np.mean(pred-saved['true'])),
   gate_weight_quantiles=np.quantile(gate,[0,.1,.5,.9,1]),gate_weight_mean=float(gate.mean()),
   source_support_mass_quantiles=np.quantile(mass,[0,.1,.5,.9,1]),
   interpretation='Postfixedresult descriptiveablation only; not a newselectioncriterion or primarycohort.')
archive=run/'predicciones_red_sin_compuerta_held_Train.npz'
assert not archive.exists()
np.savez_compressed(archive,row=saved['row'],timestamp=saved['timestamp'],true=saved['true'],**predictions)
report=dict(complete=True,scope='Postfixedresult familydiagnostic. No refit, selectionchange, extraqueries or labelchanges.',
 metrics={kind+'_graph':summary[kind]['rawgraph_with_reference_fallback_metrics'] for kind in summary},
 diagnostics=summary,predictions_sha256=sha(archive),primary_hypotheses_unchanged=True,
 above131_06_results_preserved=True,Validation=False,Test=False,source_sha256=sha(__file__))
write(run/'diagnostico_sin_compuerta.json',report);print((run/'diagnostico_sin_compuerta.json').read_text(encoding='utf-8'),flush=True)
