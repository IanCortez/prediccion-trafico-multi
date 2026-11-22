from base_replanteo import *
p=OUT/'entrenar_fases_referencia.py';s=p.read_text(encoding='utf-8-sig');s=s.replace("assert parent_sha==head['parent_sha256']","assert parent_sha==head['parent_sha256']\n    assert head['features']==names==ck['features']")
s=s.replace("torch.nn.utils.clip_grad_norm_(model.parameters(),1)\n        opt.step()","if step==first_step+1:assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)\n        torch.nn.utils.clip_grad_norm_(model.parameters(),1)\n        opt.step()")
s=s.replace("qualifies_for_preservation=metric(v,pred)['mae_s']<131.06 or metric(v,graph_pred)['mae_s']<131.06","new_primary_best=metric(v,pred)['mae_s']<metric(v,reference_pred)['mae_s'],physical_parameters=sum(p.numel() for p in model.parameters()),qualifies_for_preservation=metric(v,pred)['mae_s']<131.06 or metric(v,graph_pred)['mae_s']<131.06")
compile(s,str(p),'exec');p.write_text(s,encoding='utf-8')
policy=read(OUT/'politica_conservacion.json');policy['best_model_always_preserved']=True;policy['note']='Every complete candidate below131.06s is preserved; the strongest verified model remains protected in all cases.';write(OUT/'politica_conservacion.json',policy)
print('Final integrated trainer syntax passed; reference and first-update gradient guards enabled.')

