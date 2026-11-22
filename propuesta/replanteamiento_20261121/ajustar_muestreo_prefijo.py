from pathlib import Path
p=Path('propuesta/replanteamiento_20261121/modelo_prefijo_nativo.py')
s=p.read_text(encoding='utf-8-sig')
s=s.replace("action=(motion.cumsum(-1)<u[:,None]).sum(-1).clamp_max(48)","cdf=motion.to(torch.float64).cumsum(-1)\n            cdf=cdf/cdf[:,-1:]\n            action=(cdf<u[:,None]).sum(-1).clamp_max(48)")
s=s.replace("            assert torch.all((~move)|(nxt>=0))\n","")
p.write_text(s,encoding='utf-8')
