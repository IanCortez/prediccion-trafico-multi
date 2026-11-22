from base_replanteo import *
access=deny_test();meta=dict(np.load(DATA/'validation_x.npz'));old=np.load(BEST/'validacion.npz');y=old['true'].astype(float);x=(meta['d_lon']-meta['o_lon'])*111195*np.cos(np.deg2rad(41.16));yy=(meta['d_lat']-meta['o_lat'])*111195;distance=np.hypot(x,yy);report={}
for name,path,key in [('reference',BEST/'validacion.npz','pred'),('spatial_no_heading',OUT/'espacial_semimarkov_s0_v2/validacion.npz','graph_with_support_fallback'),('spatial_heading',OUT/'espacial_direccional_s0/validacion.npz','graph_with_support_fallback'),('spatial_count_prior',OUT/'espacial_prior_cf_s0/validacion.npz','graph_with_support_fallback')]:
 z=np.load(path);assert np.array_equal(z['row'],old['row']);p=z[key].astype(float);r=p-y;groups=[]
 for lo,hi in [(200,500),(500,1000),(1000,2000),(2000,4000),(4000,20000)]:
  keep=(distance>=lo)&(distance<hi)
  if keep.any():groups.append(dict(OD_m=[lo,hi],n=int(keep.sum()),mae_s=float(np.abs(r[keep]).mean()),mean_signed_error_s=float(r[keep].mean()),median_signed_error_s=float(np.median(r[keep]))))
 report[name]=dict(metric=metric(y,p),mean_prediction_s=float(p.mean()),mean_true_s=float(y.mean()),mean_signed_error_s=float(r.mean()),median_signed_error_s=float(np.median(r)),mean_prediction_minus_reference_s=float((p-old['pred']).mean()),by_OD_distance=groups)
write(OUT/'diagnostico_sesgo_espacial.json',dict(scope='Same complete reused Validation; descriptive diagnosis only, no cohort changes',results=report,caveat='Mean signed error need not be zero for a conditional-median predictor. Do not force mean calibration or use actual duration to gate predictions.',Test_access=False));print(json.dumps({k:{f:v[f] for f in ['mean_prediction_s','mean_signed_error_s','median_signed_error_s','mean_prediction_minus_reference_s']} for k,v in report.items()}))

