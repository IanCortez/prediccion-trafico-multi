from base_replanteo import *
p=OUT/'entrenar_atencion_subtrayectos.py';s=p.read_text()
s=s.replace('import torch,shutil,gc','import torch,shutil,gc,argparse\nap=argparse.ArgumentParser();ap.add_argument("--resume",action="store_true");args=ap.parse_args()')
s=s.replace("run=OUT/'atencion_subtrayectos_s0';assert not run.exists();run.mkdir();(run/'fuentes').mkdir()",
 "run=OUT/'atencion_subtrayectos_s0'\nif not args.resume:\n assert not run.exists();run.mkdir();(run/'fuentes').mkdir()\nelse:assert run.exists() and not (run/'resultado.json').exists()")
s=s.replace("for name in sources:shutil.copy2(OUT/name,run/'fuentes'/name)",
 "for name in sources:\n if args.resume:assert sha(OUT/name)==sha(run/'fuentes'/name)\n else:shutil.copy2(OUT/name,run/'fuentes'/name)")
s=s.replace("shutil.copy2(folder/'constructor_tokens.npz',run/'constructor_tokens.npz')",
 "if args.resume:assert sha(run/'constructor_tokens.npz')==normal['constructor_sha256']\nelse:shutil.copy2(folder/'constructor_tokens.npz',run/'constructor_tokens.npz')")
s=s.replace("write(run/'protocolo.json',config)","if args.resume:assert read(run/'protocolo.json')==config\nelse:write(run/'protocolo.json',config)")
s=s.replace("opt=torch.optim.AdamW(model.branch_parameters(),lr=.001,weight_decay=.0003);history=[]\nfor epoch in range(1,7):",
 """opt=torch.optim.AdamW(model.branch_parameters(),lr=.001,weight_decay=.0003);history=[];first_epoch=1
if args.resume:
 recovery=torch.load(run/'ultima_reanudacion.pt',map_location='cpu',weights_only=False)
 assert recovery['config']==config
 check=model.load_state_dict(recovery['branch_state'],strict=False)
 assert not check.unexpected_keys and all(k.startswith('parent.') for k in check.missing_keys)
 opt.load_state_dict(recovery['optimizer']);history=recovery['history'];first_epoch=recovery['epoch']+1
 torch.set_rng_state(recovery['torch_rng']);torch.cuda.set_rng_state_all(recovery['cuda_rng'])
for epoch in range(first_epoch,7):""")
s=s.replace("epoch=epoch,config=config,history=history),run/'ultima_reanudacion.pt')",
 "epoch=epoch,config=config,history=history,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),run/'ultima_reanudacion.pt')")
assert 'first_epoch=1' in s and 'recovery[' in s
p.write_text(s)
print('Recovery support prepared before firsttraining execution.')
