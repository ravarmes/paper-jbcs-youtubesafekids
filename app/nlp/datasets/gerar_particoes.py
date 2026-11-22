"""
Gera e grava as partições do corpus em data/particoes/.

Por que gravar a partição em vez de recalculá-la a cada script: gravada uma vez,
ela é a mesma para todos os scripts e rodadas (busca, treino final, baselines,
LLM, análises) e é publicada junto do corpus, para reprodução.

Arquivos (uma linha por frase, na ordem do corpus):

  particao_video.csv -- partição PRINCIPAL, agrupada por vídeo
      id_sentenca;video_id;particao;fold_k5;fold_k2;quase_duplicata
      * teste retido: ~20% dos vídeos (StratifiedGroupKFold, 5 grupos, o
        primeiro como teste, semente 42);
      * folds de CV (k=5 e k=2, usado na triagem da busca): StratifiedGroupKFold
        sobre o conjunto de desenvolvimento, semente 42. Toda frase de um vídeo cai
        no mesmo fold.
      A estratificação usa os rótulos do corpus v1.
      quase_duplicata = 1 marca as frases dos grupos de quase-duplicatas
      (app/nlp/datasets/quase_duplicatas.py). A divisão em si é calculada sobre o
      corpus inteiro e não depende da marca; na rodada v2, essas frases ficam fora
      dos experimentos (experimento.carregar_particao_video).

  particao_frase.csv -- a divisão da versão submetida, por frase (secundária)
      id_sentenca;video_id;particao;fold_k5;fold_k2;quase_duplicata
      Reproduz o código anterior: StratifiedKFold(5, semente 42) sobre o corpus, o
      primeiro fold como teste; folds de CV por StratifiedKFold sobre o
      desenvolvimento.

Uso:
    python -m app.nlp.datasets.gerar_particoes
    python -m app.nlp.datasets.gerar_particoes --forcar   # sobrescreve
Sem --forcar, um arquivo existente é mantido; só a coluna quase_duplicata é
acrescentada (ou conferida), sem mexer na divisão.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp import experimento
from app.nlp.datasets.prepare_data_toxicity import load_toxicity_corpus
from app.nlp.datasets import quase_duplicatas

SEMENTE = 42
N_GRUPOS_TESTE = 5   # 1/5 dos vídeos no teste
CLASSES = {0: 'Nenhuma', 1: 'Leve', 2: 'Severa'}


def _folds(indices_dev, y, grupos, k, agrupado):
    """Número do fold (1..k) de cada posição de `indices_dev`."""
    fold = np.zeros(len(indices_dev), dtype=int)
    if agrupado:
        divisor = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEMENTE)
        partes = divisor.split(indices_dev, y[indices_dev], groups=grupos[indices_dev])
    else:
        divisor = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEMENTE)
        partes = divisor.split(indices_dev, y[indices_dev])
    for f, (_, val) in enumerate(partes, 1):
        fold[val] = f
    return fold


def _montar(df, y, grupos, agrupado):
    idx = np.arange(len(df))
    if agrupado:
        divisor = StratifiedGroupKFold(n_splits=N_GRUPOS_TESTE, shuffle=True,
                                       random_state=SEMENTE)
        dev, teste = next(divisor.split(idx, y, groups=grupos))
    else:
        divisor = StratifiedKFold(n_splits=N_GRUPOS_TESTE, shuffle=True,
                                  random_state=SEMENTE)
        dev, teste = next(divisor.split(idx, y))

    particao = np.array(['treino'] * len(df), dtype=object)
    particao[teste] = 'teste'
    fold_k5 = np.full(len(df), '', dtype=object)
    fold_k2 = np.full(len(df), '', dtype=object)
    fold_k5[dev] = _folds(dev, y, grupos, 5, agrupado).astype(str)
    fold_k2[dev] = _folds(dev, y, grupos, 2, agrupado).astype(str)

    return pd.DataFrame({
        'id_sentenca': df['id_sentenca'].values,
        'video_id': grupos,
        'particao': particao,
        'fold_k5': fold_k5,
        'fold_k2': fold_k2,
    })


def _relatar(nome, part, rotulos):
    print(f"\n=== {nome} ===")
    qd = part['quase_duplicata'].astype(int) == 1
    print(f"  quase-duplicatas: {qd.sum()} frases "
          f"(treino {(qd & (part['particao'] == 'treino')).sum()}, "
          f"teste {(qd & (part['particao'] == 'teste')).sum()}; regra: "
          f"{quase_duplicatas.descrever_regra()})")
    for p in ('treino', 'teste'):
        m = part['particao'] == p
        n_vid = part.loc[m, 'video_id'].nunique()
        print(f"  {p:<6}: {m.sum():>4} frases, {n_vid:>3} vídeos")
    comp = set(part.loc[part['particao'] == 'treino', 'video_id']) & \
        set(part.loc[part['particao'] == 'teste', 'video_id'])
    print(f"  vídeos em treino e teste: {len(comp)}")
    for nome_rot, y in rotulos.items():
        for p in ('treino', 'teste'):
            m = (part['particao'] == p).values
            dist = pd.Series(y[m]).map(CLASSES).value_counts()
            pct = ' / '.join(f"{c} {100 * dist.get(c, 0) / m.sum():.1f}%"
                             for c in CLASSES.values())
            print(f"  [{nome_rot:<8}] {p:<6}: {pct}")


def main():
    ap = argparse.ArgumentParser(description='Gera as partições do corpus')
    ap.add_argument('--forcar', action='store_true', help='sobrescreve arquivos existentes')
    args = ap.parse_args()

    df = load_toxicity_corpus()
    if 'ID' not in df.columns:
        raise SystemExit("O corpus não tem a coluna de vídeo (ID/video_id).")
    y = df['label'].values
    grupos = df['ID'].astype(str).values

    rotulos = {experimento.rotulagem(): y}
    qd = quase_duplicatas.marcar(df['texto'].tolist()).astype(int)

    experimento.PARTICOES_DIR.mkdir(parents=True, exist_ok=True)
    for arquivo, agrupado, nome in (
        (experimento.ARQ_PARTICAO_VIDEO, True, 'Partição por vídeo (principal)'),
        (experimento.ARQ_PARTICAO_FRASE, False, 'Partição por frase (versão submetida)'),
    ):
        if arquivo.exists() and not args.forcar:
            print(f"\n{arquivo.name} já existe; mantido (use --forcar para regerar).")
            part = pd.read_csv(arquivo, sep=';', encoding='utf-8-sig', dtype=str)
            if (len(part) != len(df)
                    or not (part['id_sentenca'].astype(int).values == df['id_sentenca'].values).all()
                    or not (part['video_id'].values == grupos).all()):
                raise SystemExit(f"{arquivo.name} não corresponde ao corpus atual "
                                 "(tamanho, id_sentenca ou vídeo); regere com --forcar.")
            if 'quase_duplicata' not in part.columns:
                part['quase_duplicata'] = qd
                part.to_csv(arquivo, sep=';', index=False, encoding='utf-8-sig')
                print(f"Coluna quase_duplicata acrescentada a {arquivo.name}")
            elif not (part['quase_duplicata'].astype(int).values == qd).all():
                raise SystemExit(f"{arquivo.name}: quase_duplicata não bate com a regra atual")
        else:
            part = _montar(df, y, grupos, agrupado)
            part['quase_duplicata'] = qd
            part.to_csv(arquivo, sep=';', index=False, encoding='utf-8-sig')
            print(f"\nGravado: {arquivo}")
        _relatar(nome, part, rotulos)


if __name__ == '__main__':
    main()
