"""
Avalia o Detoxify multilíngue (Hanu e Unitary, 2020) no corpus, sem ajuste fino.

O Detoxify devolve, por frase, probabilidades independentes para atributos do desafio
Jigsaw (toxicity, severe_toxicity, obscene, identity_attack, insult, threat,
sexual_explicit). Nenhum deles é a escala de três níveis do artigo, então há um
mapeamento, e todo parâmetro do mapeamento é escolhido **só no treino** da partição por
vídeo; o teste é usado uma vez, para reportar.

Três leituras, da menos para a mais informada pelos nossos rótulos:

  binaria_padrao   -> tóxico se toxicity >= 0,5 (o limiar convencional do modelo);
                      avaliado na tarefa binária tóxico x não tóxico.
  binaria_treino   -> o mesmo, com o limiar que maximiza o F1 macro binário no treino.
  tres_limiares    -> Nenhuma / Leve / Severa por dois cortes t1 < t2 sobre toxicity,
                      escolhidos no treino (grade sobre os quantis do treino).
  tres_logistica   -> regressão logística multinomial sobre os atributos do Detoxify,
                      ajustada no treino. É uma sonda linear sobre as saídas congeladas:
                      diz quanto da escala de três níveis já está nas saídas do modelo.

Os escores das frases ficam em cache (CSV), para que o mapeamento possa ser refeito sem
rodar o modelo de novo.

Dependências: `detoxify==0.5.3` e `sentencepiece` (no requirements.txt). O checkpoint
multilíngue (multilingual_debiased-0b549669.ckpt, 1,1 GB) é baixado do repositório do
Detoxify no primeiro uso, para o cache do torch.hub; ver docs/external-models.md.

Uso (de src/):
    python app/nlp/evaluation/scripts/avaliar_detoxify.py
    python app/nlp/evaluation/scripts/avaliar_detoxify.py --cpu
    python app/nlp/evaluation/scripts/avaliar_detoxify.py --so-mapeamento   # usa o cache
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
)
sys.path.insert(0, project_root)

from app.nlp import experimento  # noqa: E402

CORPUS = os.path.join(project_root, 'data', 'corpus_toxicidade_publico.csv')
ROTULOS = {'Nenhuma': 0, 'Leve': 1, 'Severa': 2}
VARIANTE = 'multilingual'
SEMENTE = 42


def carregar():
    corpus = pd.read_csv(CORPUS, sep=';', encoding='utf-8')
    # Pelo carregador da rodada: na v2, as quase-duplicatas vêm como 'excluida' e ficam
    # fora do treino e do teste.
    part = experimento.carregar_particao_video()
    df = corpus.merge(part[['id_sentenca', 'particao']], on='id_sentenca', how='inner')
    if len(df) != len(corpus):
        raise SystemExit(f'partição cobre {len(df)} de {len(corpus)} frases')
    df['y'] = df['tox'].map(ROTULOS)
    if df['y'].isna().any():
        raise SystemExit('rótulo desconhecido no corpus')
    df['y'] = df['y'].astype(int)
    return df


def pontuar(textos, cpu=False, lote=32):
    from detoxify import Detoxify
    import torch
    dispositivo = 'cpu' if cpu or not torch.cuda.is_available() else 'cuda'
    modelo = Detoxify(VARIANTE, device=dispositivo)
    saidas = {}
    for i in range(0, len(textos), lote):
        r = modelo.predict(list(textos[i:i + lote]))
        for k, v in r.items():
            saidas.setdefault(k, []).extend(np.atleast_1d(v).tolist())
    import detoxify
    meta = {'variante': VARIANTE, 'dispositivo': dispositivo,
            'versao_detoxify': getattr(detoxify, '__version__', 'desconhecida'),
            'versao_torch': torch.__version__}
    return pd.DataFrame(saidas), meta


def _f1_bin(y_bin, pred):
    return float(f1_score(y_bin, pred, average='macro'))


def limiar_binario(score, y_bin):
    candidatos = np.unique(np.quantile(score, np.linspace(0.01, 0.99, 197)))
    melhor = max(candidatos, key=lambda t: _f1_bin(y_bin, (score >= t).astype(int)))
    return float(melhor)


def dois_limiares(score, y):
    q = np.unique(np.quantile(score, np.linspace(0.02, 0.98, 49)))
    melhor, valor = (None, None), -1.0
    for i, t1 in enumerate(q):
        for t2 in q[i + 1:]:
            pred = np.where(score >= t2, 2, np.where(score >= t1, 1, 0))
            v = float(f1_score(y, pred, average='macro'))
            if v > valor:
                melhor, valor = (float(t1), float(t2)), v
    return melhor


def metricas(y, pred, n_classes):
    rotulos = list(range(n_classes))
    return {
        'f1_macro': round(float(f1_score(y, pred, average='macro', labels=rotulos)), 4),
        'f1_por_classe': [round(float(x), 4)
                          for x in f1_score(y, pred, average=None, labels=rotulos)],
        'matriz_confusao': confusion_matrix(y, pred, labels=rotulos).tolist(),
    }


def mapear(df, atributos):
    tr = df['particao'] == 'treino'
    te = df['particao'] == 'teste'
    y_tr, y_te = df.loc[tr, 'y'].to_numpy(), df.loc[te, 'y'].to_numpy()
    yb_tr, yb_te = (y_tr > 0).astype(int), (y_te > 0).astype(int)
    s_tr, s_te = df.loc[tr, 'toxicity'].to_numpy(), df.loc[te, 'toxicity'].to_numpy()

    res = {}
    pred = (s_te >= 0.5).astype(int)
    res['binaria_padrao'] = {'limiar': 0.5, 'teste': metricas(yb_te, pred, 2),
                             'y_pred_teste': pred.tolist()}

    t = limiar_binario(s_tr, yb_tr)
    pred = (s_te >= t).astype(int)
    res['binaria_treino'] = {
        'limiar': round(t, 6),
        'treino_f1_macro': round(_f1_bin(yb_tr, (s_tr >= t).astype(int)), 4),
        'teste': metricas(yb_te, pred, 2), 'y_pred_teste': pred.tolist()}

    t1, t2 = dois_limiares(s_tr, y_tr)
    pred_tr = np.where(s_tr >= t2, 2, np.where(s_tr >= t1, 1, 0))
    pred = np.where(s_te >= t2, 2, np.where(s_te >= t1, 1, 0))
    res['tres_limiares'] = {
        'limiares': [round(t1, 6), round(t2, 6)],
        'treino_f1_macro': round(float(f1_score(y_tr, pred_tr, average='macro')), 4),
        'teste': metricas(y_te, pred, 3), 'y_pred_teste': pred.tolist()}

    X_tr = df.loc[tr, atributos].to_numpy()
    X_te = df.loc[te, atributos].to_numpy()
    esc = StandardScaler().fit(X_tr)
    lr = LogisticRegression(max_iter=2000, random_state=SEMENTE)
    lr.fit(esc.transform(X_tr), y_tr)
    pred_tr = lr.predict(esc.transform(X_tr))
    pred = lr.predict(esc.transform(X_te))
    res['tres_logistica'] = {
        'atributos': atributos,
        'treino_f1_macro': round(float(f1_score(y_tr, pred_tr, average='macro')), 4),
        'teste': metricas(y_te, pred, 3), 'y_pred_teste': pred.tolist()}

    res['_ids_teste'] = df.loc[te, 'id_sentenca'].astype(int).tolist()
    res['_y_teste'] = y_te.tolist()
    return res


def main():
    p = argparse.ArgumentParser(description='Avalia o Detoxify multilíngue no corpus')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--so-mapeamento', action='store_true',
                   help='não roda o modelo; usa os escores em cache')
    a = p.parse_args()

    df = carregar()
    pasta = experimento.dir_resultados('detoxify')
    cache = pasta / 'escores_detoxify.csv'

    if a.so_mapeamento:
        if not cache.exists():
            raise SystemExit(f'sem cache em {cache}; rode sem --so-mapeamento')
        escores = pd.read_csv(cache, sep=';')
        meta = json.loads((pasta / 'escores_meta.json').read_text(encoding='utf-8'))
    else:
        escores, meta = pontuar(df['frase'].astype(str).tolist(), cpu=a.cpu)
        escores.insert(0, 'id_sentenca', df['id_sentenca'].to_numpy())
        escores.to_csv(cache, sep=';', index=False)
        (pasta / 'escores_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    atributos = [c for c in escores.columns if c != 'id_sentenca']
    if 'toxicity' not in atributos:
        raise SystemExit(f'saída do Detoxify sem "toxicity": {atributos}')
    df = df.merge(escores, on='id_sentenca', how='inner')
    resultado = {
        'rodada': experimento.descricao(),
        'modelo': meta,
        'citacao': 'hanu2020detoxify',
        'protocolo': 'parâmetros do mapeamento escolhidos só no treino da partição por vídeo',
        'n_treino': int((df['particao'] == 'treino').sum()),
        'n_teste': int((df['particao'] == 'teste').sum()),
        'resultados': mapear(df, atributos),
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    saida = pasta / f'detoxify_{ts}.json'
    saida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding='utf-8')
    for nome, r in resultado['resultados'].items():
        if not nome.startswith('_'):
            print(f"{nome:16s} teste F1 macro = {r['teste']['f1_macro']}")
    print(f'-> {saida}')


if __name__ == '__main__':
    main()
