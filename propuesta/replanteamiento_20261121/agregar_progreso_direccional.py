from pathlib import Path
p=Path('E:/UTEC/Tesis/TesisIan/propuesta/replanteamiento_20261121/entrenar_direccional.py');assert not (p.parent/'espacial_direccional_s0').exists();s=p.read_text(encoding='utf-8-sig');old="ix=torch.tensor(use[first:first+16],device='cuda');n=len(ix)";new="if indices is None and first%2048==0:print(json.dumps(dict(stage='full_validation_prediction',completed_rows=first,total_rows=len(use))),flush=True)\n   ix=torch.tensor(use[first:first+16],device='cuda');n=len(ix)";assert old in s;s=s.replace(old,new);p.write_text(s,encoding='utf-8')

