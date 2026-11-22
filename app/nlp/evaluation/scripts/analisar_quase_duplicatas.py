"""
Grupos de quase-duplicatas: descrição e efeito no teste (revisão do JBCS; parecer B1).

As frases dos grupos de quase-duplicatas (regra em app/nlp/datasets/quase_duplicatas.py)
terminam com a mesma sequência de palavras que outras frases de vídeos diferentes. A
partição por vídeo não as separa, e um modelo pode classificá-las pela terminação.

Este script, sem treinar nada e sem torch:
  1. descreve os grupos no corpus publicado: frases, terminações, vídeos, rótulos,
     lotes, frases sem palavrão, lado da partição por vídeo;
  2. mede, nas predições gravadas da rodada, o F1 macro do teste inteiro e do teste sem
     as frases de cada regra (a principal e as vizinhas), para o BERTimbau (5 sementes,
     argmax) e para o SVM linear ajustado, e a acurácia nas frases marcadas e nas demais
     frases tóxicas.
     - Na rodada v1, o teste ainda contém as quase-duplicatas: a diferença mede o
       quanto elas inflavam o resultado.
     - Na rodada v2, elas já estão fora; a análise mede a sensibilidade do resultado às
       regras vizinhas (frases a mais que cada regra marcaria).

Lê:
  data/corpus_toxicidade_publico.csv
  data/particoes/particao_video.csv (coluna particao original, sem a exclusão da v2)
  <resultados>/modelo_final/probabilidades_teste_*.csv  (mais recente)
  <resultados>/predicoes_teste_baselines_ajustados_*.csv (pred_svm)
Grava <resultados>/quase_duplicatas_<timestamp>.json.

Uso (de src/):
    python app/nlp/evaluation/scripts/analisar_quase_duplicatas.py           # rodada atual
    $env:YSK_ROTULAGEM='v1'; python app/nlp/evaluation/scripts/analisar_quase_duplicatas.py
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
from app.nlp.datasets import quase_duplicatas as qd  # noqa: E402

CLASSES = ['Nenhuma', 'Leve', 'Severa']


def mais_recente(padrao):
    arqs = sorted(glob.glob(padrao), key=os.path.getmtime)
    if not arqs:
        raise FileNotFoundError(padrao)
    return arqs[-1]


def f1m(y, p):
    return float(f1_score(y, p, average='macro', labels=[0, 1, 2], zero_division=0))


def nome_regra(n, minimo):
    return f'{n}_palavras_min_{minimo}'


def main():
    res = experimento.dir_resultados()
    corpus = pd.read_csv(os.path.join(project_root, 'data', 'corpus_toxicidade_publico.csv'), sep=';')
    part = pd.read_csv(experimento.ARQ_PARTICAO_VIDEO, sep=';', encoding='utf-8-sig',
                       dtype={'video_id': str})
    corpus = corpus.merge(part[['id_sentenca', 'particao']], on='id_sentenca', validate='one_to_one')

    regras = [(qd.N_PALAVRAS, qd.MIN_FRASES), *qd.REGRAS_SENSIBILIDADE]
    for n, minimo in regras:
        corpus[nome_regra(n, minimo)] = qd.marcar(corpus['frase'].tolist(), n, minimo)
    principal = nome_regra(qd.N_PALAVRAS, qd.MIN_FRASES)
    if not (corpus[principal].astype(int).values == part['quase_duplicata'].astype(int).values).all():
        raise SystemExit('a regra principal não bate com a coluna quase_duplicata da partição')

    g = corpus[corpus[principal]]
    rotulos_por_fim = g.groupby(g['frase'].map(qd.terminacao))['tox']
    toxicas = corpus['tox'] != 'Nenhuma'
    sem_palavrao = toxicas & (corpus['tem_palavrao'] == 0)
    descricao = {
        'regra': qd.descrever_regra(),
        'n_frases': int(len(g)),
        'n_terminacoes': int(g['frase'].map(qd.terminacao).nunique()),
        'terminacoes_rotulo_unico': int((rotulos_por_fim.nunique() == 1).sum()),
        'maior_grupo': int(rotulos_por_fim.size().max()),
        'n_videos': int(g['video_id'].nunique()),
        'distribuicao': {c: int((g['tox'] == c).sum()) for c in CLASSES},
        'por_lote': {str(k): int(v) for k, v in g['lote'].value_counts().sort_index().items()},
        'por_particao': {k: int(v) for k, v in g['particao'].value_counts().items()},
        'toxicas_sem_palavrao_no_grupo': int((g['tem_palavrao'] == 0).sum()),
        'toxicas_sem_palavrao_no_corpus': int(sem_palavrao.sum()),
        'toxicas_no_corpus': int(toxicas.sum()),
        'frases_corpus': int(len(corpus)),
        'videos_corpus': int(corpus['video_id'].nunique()),
        'regras_vizinhas': {nome_regra(n, m): {
            'n_frases': int(corpus[nome_regra(n, m)].sum()),
            'n_nenhuma': int((corpus[nome_regra(n, m)] & (corpus['tox'] == 'Nenhuma')).sum())}
            for n, m in qd.REGRAS_SENSIBILIDADE},
    }

    arq_prob = mais_recente(str(res / 'modelo_final' / 'probabilidades_teste_*.csv'))
    arq_svm = mais_recente(str(res / 'predicoes_teste_baselines_ajustados_*.csv'))
    prob = pd.read_csv(arq_prob, sep=';')
    svm = pd.read_csv(arq_svm, sep=';')[['id_sentenca', 'pred_svm']]
    df = (prob.merge(corpus, on='id_sentenca', how='left', validate='one_to_one')
              .merge(svm, on='id_sentenca', how='left', validate='one_to_one'))
    sementes = [int(re.match(r'p_nenhuma_s(\d+)$', c).group(1))
                for c in prob.columns if c.startswith('p_nenhuma_s')]
    for s in sementes:
        df[f'pred_s{s}'] = df[[f'p_{c.lower()}_s{s}' for c in CLASSES]].to_numpy().argmax(axis=1)

    def bloco(d):
        v = np.array([f1m(d['rotulo'], d[f'pred_s{s}']) for s in sementes])
        return {'n_frases': int(len(d)), 'n_videos': int(d['video_id'].nunique()),
                'bertimbau_f1_media': float(v.mean()),
                'bertimbau_f1_dp': float(v.std(ddof=1)) if len(v) > 1 else None,
                'svm_ajustado_f1': f1m(d['rotulo'], d['pred_svm'].astype(int))}

    def acuracia(d):
        if len(d) == 0:
            return None
        return {'n_frases': int(len(d)),
                'bertimbau_media': float(np.mean([(d['rotulo'] == d[f'pred_s{s}']).mean()
                                                  for s in sementes])),
                'svm_ajustado': float((d['rotulo'] == d['pred_svm'].astype(int)).mean())}

    teste = {'inteiro': bloco(df)}
    for n, m in regras:
        col = nome_regra(n, m)
        teste[f'sem_{col}'] = {**bloco(df[~df[col]]), 'frases_retiradas': int(df[col].sum())}
    marcadas = df[df[principal]]
    demais_toxicas = df[~df[principal] & (df['rotulo'] > 0)]

    saida = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'script': 'app/nlp/evaluation/scripts/analisar_quase_duplicatas.py',
        'experimento': experimento.descricao(),
        'fontes': {'probabilidades': os.path.basename(arq_prob), 'svm_ajustado': os.path.basename(arq_svm)},
        'descricao_grupos': descricao,
        'teste': teste,
        'acuracia_quase_duplicatas_teste': acuracia(marcadas),
        'acuracia_demais_toxicas_teste': acuracia(demais_toxicas),
        'nota': ('Na v1 o teste contém as quase-duplicatas; na v2 elas já estão fora, e as '
                 'linhas sem_<regra vizinha> medem só as frases a mais daquela regra.'),
    }
    arq = res / f"quase_duplicatas_{saida['timestamp']}.json"
    arq.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: saida[k] for k in ('descricao_grupos', 'teste', 'acuracia_quase_duplicatas_teste',
                                            'acuracia_demais_toxicas_teste')}, indent=1, ensure_ascii=False))
    print(f'-> {arq}')


if __name__ == '__main__':
    main()
