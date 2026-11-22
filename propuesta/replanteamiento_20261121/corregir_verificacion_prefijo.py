from pathlib import Path
p=Path('propuesta/replanteamiento_20261121/verificar_prefijo_nativo.py')
s=p.read_text(encoding='utf-8-sig')
s=s.replace("loss=-motion[:,0,0].log().mean()","move_index=int(np.flatnonzero((offsets==[1,0]).all(-1))[0])+1\nloss=-motion[:,0,move_index].log().mean()")
s=s.replace("datos_semi_markov/manifiesto.json","datos_semi_markov/protocolo.json")
p.write_text(s,encoding='utf-8')
