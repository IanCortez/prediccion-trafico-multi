"""Inventory additional historical parents from binary metadata only; never decode GPS."""
from base_replanteo import *
from collections import Counter
access=deny_test();start=time.time();tr=dict(np.load(DATA/'train_x.npz'))
manifest=read(DATA/'manifiesto.json');source=ROOT/'tesis/porto/porto/train.csv'
assert source.exists()
known=set(map(int,tr['row']));pilot=read(OUT/'piloto_transferencia_intervalos_Train_s0/verificacion_publica.json')['cutoff']
primary=int(manifest['validation']['start_timestamp'])
counts=Counter();examples=[];stamps=[];raw_n=0
with source.open('rb') as f:
 header=next(f).strip()
 expected=b'"TRIP_ID","CALL_TYPE","ORIGIN_CALL","ORIGIN_STAND","TAXI_ID","TIMESTAMP","DAY_TYPE","MISSING_DATA","POLYLINE"'
 assert header.lstrip(bytes([239,187,191]))==expected
 for row,line in enumerate(f):
  raw_n+=1
  if row in known:
   counts['known_primary_Train_rows_skipped_before_parse']+=1;continue
  # Parse only ASCII binary timestamp/missing metadata, never UTF8/CSV/JSON or GPS payload.
  fields=line.split(b',',8);assert len(fields)==9
  stamp=int(fields[5].strip(b'"'));missing=fields[7].strip(b'"')==b'True'
  counts['outside_primary_Train_metadata_rows']+=1
  if missing:counts['outside_missing_flag']+=1;continue
  for name,boundary in [('pilot',pilot),('primary',primary)]:
   if stamp<boundary-86400:counts[name+'_potential_parents_before_24h_buffer']+=1
  if stamp<primary-86400:
   stamps.append(stamp)
   if len(examples)<8:examples.append(dict(rawrow=row,timestamp=stamp))
assert raw_n==manifest['counts']['raw']
assert counts['known_primary_Train_rows_skipped_before_parse']==len(tr['row'])
result=dict(complete=True,scope='Additional-source availability inventory only, not a fitted model, outcome or new primarycohort.',
 binary_metadata_only=True,GPS_payloads_decoded=0,original_primary_rows_labels_and_split_unchanged=True,
 counts=counts,raw_rows=raw_n,examples_metadata=examples,minimum_potential_timestamp=min(stamps),maximum_potential_timestamp=max(stamps),
 important_limit='Potentialparents are not yet authorized as valid supervision by this inventory. Any later constructor must select earlyTrain by timestamp before GPSdecode, then verify sourcearrival cutoff and usable intervals. Buffer alone does not prove arrival.',
 distinction='Priorregionaltransfer split existingprimary atlongitude−8.62; it did not inventory outside-primary rawTrainparents.',
 source_sha256_previously_verified=manifest['source_sha256'],source_file_bytes=source.stat().st_size,
 code_sha256=sha(__file__),Test_files_opened=False,no_model_or_prediction_claim=True,actual_data_files_accessed=sorted(access),seconds=time.time()-start)
write(OUT/'inventario_fuentes_adicionales.json',result)
print(json.dumps(read(OUT/'inventario_fuentes_adicionales.json')),flush=True)
