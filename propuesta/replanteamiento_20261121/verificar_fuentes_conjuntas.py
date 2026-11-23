from base_replanteo import *
p=OUT/'entrenar_subtrayectos_conjuntos.py';s=p.read_text()
s=s.replace('"""Fixed-six-epoch learned subtrajectory expert, retaining the protected parent hierarchy."""',
 '"""Fixed24epoch jointlytrained fresh30component OD/time/source mixture; no pretrainedneuralweights."""')
s=s.replace(" parent_id=reference['id'],parent_sha256=reference['model_sha256'],parent_validation=reference['validation'],",
 " comparison_reference_id=reference['id'],comparison_reference_sha256=reference['model_sha256'],reference_validation=reference['validation'],\n constructor_sha256={name:sha(OUT/'historica_fresca_s0'/name) for name in ['constructor_base.pt','constructor_historia.npz','constructor_mapa.npz']},")
s=s.replace("model=JointSubtrajectoryAttention(parent,constructor['mean'],constructor['std'],constructor['time_anchor'],'individual').cuda()",
 """geometry_proof=read(OUT/'verificacion_geometria_tokens_subtrayectos.json')
assert geometry_proof['passed'] and geometry_proof['token_sha256']==sha(OUT/'tokens_subtrayectos.py')
model=JointSubtrajectoryAttention(parent,constructor['mean'],constructor['std'],constructor['time_anchor'],'individual')
initial=hashlib.sha256()
for name,value in sorted(model.state_dict().items()):initial.update(name.encode());initial.update(value.detach().numpy().tobytes())
write(run/'inicializacion_fresca.json',dict(initial_state_sha256=initial.hexdigest(),all_neural_parameters_require_grad=all(p.requires_grad for p in model.parameters()),neural_pretrained_weights_loaded=False))
model=model.cuda()""")
old=" assert model.duration.backbone.output.weight.grad is not None if hasattr(model.duration.backbone.output,'weight') else any(p.grad is not None and p.grad.norm()>0 for p in model.duration.parameters())"
s=s.replace(old,
 " assert any(p.grad is not None and p.grad.norm()>0 for p in model.duration.parameters())\n assert model.head[-1].weight.grad is not None and model.head[-1].weight.grad.norm()>0\n assert all(p.requires_grad for p in model.parameters())")
marker="@torch.no_grad()\ndef predict(parent_only=False):"
new="""# Verify the actual canonical inference memory peak before committing to training.
model.eval()
with torch.no_grad():
 probe=ids[:32].repeat(64);torch.cuda.reset_peak_memory_stats();sub=tokens(s,si,probe)
 output=model(x[probe],c[probe],h[probe],sub)
 assert torch.isfinite(output[0]).all()
 write(run/'preflight_consulta_2048.json',dict(passed=True,batch=2048,GPU_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2,no_Validationmetric=True))
 del sub,output,probe
@torch.no_grad()
def predict():"""
assert marker in s;s=s.replace(marker,new)
s=s.replace("# Only the latest small branch state for interruption recovery; no duplicate parentweights.",
 "# Keep one full model+optimizer checkpoint for interruption recovery.")
s=s.replace("model_digest=sha(path);p=predict();",
 """model_digest=sha(path)
# Release Train tensors before full evaluation; Validation queries and source memory suffice.
for container in [public,histories,subqueries,indices]:container[0]=None
del x,c,h,s,si,target,ids,g,valid,order,index,opt,loss
gc.collect();torch.cuda.empty_cache()
p=predict();""")
s=s.replace("parent_unchanged=sha(reference['model_file'])==reference['model_sha256']",
 "protected_reference_unchanged=sha(reference['model_file'])==reference['model_sha256']")
p.write_text(s)
import ast
ast.parse(s,filename=p.name)
write(OUT/'fuentes_conjuntas_verificadas.json',dict(syntax_valid=True,sources={name:sha(OUT/name) for name in ['modelo_subtrayectos_conjuntos.py','entrenar_subtrayectos_conjuntos.py']},
 joint_training_real_gradient_and_canonical2048_preflight='Required before optimizerupdates; no frozenparent gradients allowed.',
 final_evaluation='Train tensors explicitly released before fullValidation; no enlargedGPUcap.'))
print('Jointfresh-model sources reviewed and finalized before launch.')
