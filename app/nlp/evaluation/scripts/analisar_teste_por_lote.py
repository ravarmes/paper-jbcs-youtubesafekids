"""
Desempenho no teste separado por lote de anotação, e época escolhida no treino final.

Revisão do JBCS (M2 e M3c):
  - o corpus v1 tem dois lotes (coluna `lote` do corpus público): o 1º, anotado antes de
    qualquer modelo; o 2º, anotado em nov/2026, depois dos resultados da versão submetida.
    Este script mede o F1 macro do teste em cada lote, para o BERTimbau (5 sementes do
    treino final, argmax) e para o SVM linear ajustado (CV aninhada), e a composição de
    cada lote no teste;
  - lê do log do treino final a época cujos pesos foram restaurados em cada semente
    (o treino guarda a melhor época no conjunto interno de desenvolvimento).

Nada é treinado; sem torch. Lê:
  data/corpus_toxicidade_publico.csv               (id_sentenca, video_id, lote)
  <resultados>/modelo_final/probabilidades_teste_*.csv  (mais recente)
  <resultados>/modelo_final/modelo_final_*.json          (sementes, para conferência)
  <resultados>/predicoes_teste_baselines_ajustados_*.csv (pred_svm)
  <resultados>/logs/final.log                            (época restaurada)
Grava <resultados>/teste_por_lote_<timestamp>.json.

Uso (de src/):
    python app/nlp/evaluation/scripts/analisar_teste_por_lote.py
"""

import glob
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp import experimento  # noqa: E402

CLASSES = ['Nenhuma', 'Leve', 'Severa']


def mais_recente(padrao):
    arqs = sorted(glob.glob(padrao), key=os.path.getmtime)
    if not arqs:
        raise FileNotFoundError(padrao)
    return arqs[-1]


def f1m(y, p):
    return float(f1_score(y, p, average='macro', labels=[0, 1, 2], zero_division=0))


def main():
    res = experimento.dir_resultados()
    corpus = pd.read_csv(os.path.join(project_root, 'data', 'corpus_toxicidade_publico.csv'),
                         sep=';', usecols=['id_sentenca', 'video_id', 'lote'])

    arq_prob = mais_recente(str(res / 'modelo_final' / 'probabilidades_teste_*.csv'))
    arq_final = mais_recente(str(res / 'modelo_final' / 'modelo_final_*.json'))
    arq_svm = mais_recente(str(res / 'predicoes_teste_baselines_ajustados_*.csv'))
    prob = pd.read_csv(arq_prob, sep=';')
    final = json.load(open(arq_final, encoding='utf-8'))
    svm = pd.read_csv(arq_svm, sep=';')[['id_sentenca', 'pred_svm']]

    df = prob.merge(corpus, on='id_sentenca', how='left', validate='one_to_one')
    if df['lote'].isna().any():
        raise SystemExit('frases do teste sem lote no corpus público')
    df = df.merge(svm, on='id_sentenca', how='left', validate='one_to_one')
    if df['pred_svm'].isna().any():
        raise SystemExit('frases do teste sem predição do SVM ajustado')

    sementes = [int(re.match(r'p_nenhuma_s(\d+)$', c).group(1))
                for c in prob.columns if c.startswith('p_nenhuma_s')]
    for s in sementes:
        cols = [f'p_{c.lower()}_s{s}' for c in CLASSES]
        df[f'pred_s{s}'] = df[cols].to_numpy().argmax(axis=1)

    # conferência: o F1 global por semente tem de bater com o do treino final
    gravados = {int(x['seed']): x['f1_macro'] for x in final['teste_por_semente']}
    conferencia = {}
    for s in sementes:
        f = f1m(df['rotulo'], df[f'pred_s{s}'])
        conferencia[s] = {'recalculado': f, 'gravado': gravados.get(s)}
        if gravados.get(s) is not None and abs(f - gravados[s]) > 1e-9:
            raise SystemExit(f'F1 da semente {s} não confere: {f} contra {gravados[s]}')

    def bloco(d):
        por_semente = {s: f1m(d['rotulo'], d[f'pred_s{s}']) for s in sementes}
        v = np.array(list(por_semente.values()))
        dist = d['rotulo'].value_counts().reindex([0, 1, 2], fill_value=0)
        return {
            'n_frases': int(len(d)),
            'n_videos': int(d['video_id'].nunique()),
            'distribuicao': {CLASSES[i]: int(dist[i]) for i in range(3)},
            'bertimbau': {
                'f1_macro_por_semente': {str(k): x for k, x in por_semente.items()},
                'f1_macro_media': float(v.mean()),
                'f1_macro_dp': float(v.std(ddof=1)) if len(v) > 1 else None,
                'f1_macro_semente_referencia': por_semente.get(int(final['semente_referencia'])),
            },
            'svm_ajustado': {'f1_macro': f1m(d['rotulo'], d['pred_svm'].astype(int))},
        }

    saida = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'script': 'app/nlp/evaluation/scripts/analisar_teste_por_lote.py',
        'experimento': experimento.descricao(),
        'fontes': {k: os.path.basename(v) for k, v in
                   (('probabilidades', arq_prob), ('modelo_final', arq_final),
                    ('svm_ajustado', arq_svm))},
        'regra': 'argmax das probabilidades; DP entre sementes com ddof=1 (o mesmo de teste_resumo)',
        'conferencia_f1_global': {str(k): v for k, v in conferencia.items()},
        'global': bloco(df),
        'por_lote': {str(int(l)): bloco(d) for l, d in df.groupby('lote')},
    }
    vids = df.groupby('video_id')['lote'].nunique()
    saida['videos_com_os_dois_lotes_no_teste'] = int((vids > 1).sum())

    # época restaurada por semente, do log do treino final
    log = res / 'logs' / 'final.log'
    if log.exists():
        # O log tem também os treinos dos folds das predições fora da amostra, que não
        # registram a semente; vale, para cada semente, a primeira época restaurada
        # depois do início do treino dela.
        por_semente, atual = {}, None
        for linha in open(log, encoding='utf-8', errors='replace'):
            m_seed = re.search(r"'seed': (\d+), 'usar_melhor_epoca'", linha)
            if m_seed:
                atual = int(m_seed.group(1))
                continue
            m_ep = re.search(r'Pesos restaurados da época (\d+)', linha)
            if m_ep and atual is not None and atual not in por_semente:
                por_semente[atual] = int(m_ep.group(1))
                atual = None
        if sorted(por_semente) == sorted(sementes):
            saida['epoca_restaurada'] = {
                'fonte': 'logs/final.log',
                'epocas_maximas': int(final['config']['epochs']),
                'por_semente': {str(s): por_semente[s] for s in sementes},
            }
        else:
            saida['epoca_restaurada'] = {'erro': f'sementes no log: {sorted(por_semente)}'}

    arq = res / f"teste_por_lote_{saida['timestamp']}.json"
    arq.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: saida[k] for k in ('por_lote', 'videos_com_os_dois_lotes_no_teste',
                                            'epoca_restaurada') if k in saida},
                     indent=1, ensure_ascii=False))
    print(f'JSON: {arq}')


if __name__ == '__main__':
    main()
