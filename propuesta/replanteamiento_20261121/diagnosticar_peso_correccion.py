from base_replanteo import *
import torch,io,math
torch.set_num_threads(2);records=[]
for name in ['espacial_prior_cf_s0','espacial_fases_referencia_s0']:
 run=OUT/name;p=run/'modelo.pt'
 for attempt in range(5):
  before=read(run/'mejor.json');st=p.stat();blob=p.read_bytes();after=read(run/'mejor.json')
  if before==after and st.st_mtime_ns==p.stat().st_mtime_ns:break
 else:raise RuntimeError('unstable snapshot')
 ck=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False);logit=float(ck['state']['mix_logit']);records.append(dict(id=name,selected=before,logit=logit,nominal_fraction=1/(1+math.exp(-logit)),checkpoint_sha256=hashlib.sha256(blob).hexdigest()))
assert not torch.cuda.is_initialized()
report=dict(records=records,current_scalar_learning_rate=.0001,interpretation='The global blend parameter was inherited from a weaker direct head. A slow scalar learning rate may retain excessive physical correction when replacing it with a stronger frozen reference. This is an optimization/calibration hypothesis, not evidence that the physical branch improves MAE.',next='After the trained model is frozen, one predeclared small global-weight calibration on the same fixed selection subset may test this hypothesis without retraining or disk CDF banks. Report the uncalibrated result separately. Do not launch an endless weight search.',code_sha256=sha(__file__),Test_access=False,no_GPU_used=True)
write(OUT/'diagnostico_peso_correccion.json',report);print(json.dumps(report))

