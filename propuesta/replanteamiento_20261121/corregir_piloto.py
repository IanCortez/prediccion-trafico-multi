from base_replanteo import *
p=OUT/'piloto_entorno_compartido.py';s=p.read_text(encoding='utf-8-sig');s=s.replace("loss.backward()\n        optimizer.step()\n        score=float(loss.detach())","loss.backward()\n        candidate_parameters=(bias.detach().clone(),raw_sigma.detach().clone())\n        optimizer.step()\n        score=float(loss.detach())");s=s.replace("best_parameters=(bias.detach().clone(),raw_sigma.detach().clone())","best_parameters=candidate_parameters");compile(s,str(p),'exec');p.write_text(s,encoding='utf-8')

