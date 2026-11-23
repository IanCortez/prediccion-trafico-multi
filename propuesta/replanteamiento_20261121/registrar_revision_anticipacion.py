from base_replanteo import *
r=dict(review_date='2026-11-22',paper='Looking Ahead and Back: Anticipatory Generative Modeling for Road-Network Trajectories',
source='https://arxiv.org/html/2610.13131v1',status='arXiv preprint v1, submitted 2026-10-09; no peer-review status established here',
verified_scope='Destination-conditioned road trajectory generation. Fixed horizon future anticipation supervises future windows only in training, then predicts its own continuation at inference.',
compatibility='Conceptual route-generation antecedent, not direct proof of OD duration accuracy. Reported Porto uses HOSER graphs/splits, not the current primary cohort.',
metrics='JSD, Hausdorff, EDR and route-fidelity/completion measures; not the target OD MAE in seconds.',
potential_idea='Use Train-only future-window targets to make generated-route choice sensitive to downstream consequences.',
local_inference='A future local adaptation would have to generate all predictive context from OD/time and its own generated prefix. True future, true length and query path remain forbidden.',
caveat='Better route fidelity need not reduce time MAE. A matched time forecast control and full fixed-cohort evaluation would be required.',
decision='No new GPU job or dataset download. Current full-trip risk and semantic-head experiments take precedence.',
source_word_budget_note='Concise paraphrase only; no extracted tables or claims of local replication.')
write(OUT/'revision_anticipacion_20261122.json',r)
marker='<!-- antecedente_anticipacion_20261122 -->'
text="""

<!-- antecedente_anticipacion_20261122 -->
Se revisó [Foresight (prepublicación, 9 de octubre de 2026)](https://arxiv.org/html/2610.13131v1): predice una continuación y la usa para escoger transiciones de ruta. Sus resultados Porto son de fidelidad y finalización de trayectorias sobre HOSER; no miden nuestro MAE OD. Puede motivar supervisión de ventanas futuras exclusivamente en Train. Una eventual adaptación deberá generar su propio futuro en consulta y compararse con un control de tiempo; no se atribuyen ganancias de duración a ese antecedente. Por ahora no se agrega otra corrida GPU ni otra descarga de datos.
"""
p=OUT/'README.md';s=p.read_text(encoding='utf-8-sig')
if marker not in s:p.write_text(s+text,encoding='utf-8')
print('Revisión crítica registrada con su alcance y fuente primaria')
