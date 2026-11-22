"""
Script de preparação de dados para Detecção de Toxicidade (TOX).
Configurado conforme metodologia:
- 80% Treino (Usado para Cross-Validation)
- 20% Teste (Reservado para avaliação final)

Formato do corpus: separador ';', UTF-8 com BOM, em duas variantes aceitas.
  - data/corpus_toxicidade_publico.csv -- o arquivo PUBLICADO, e o unico que
    acompanha o repositorio. Colunas: id_sentenca, video_id, titulo, parte, link,
    frase, n_palavras, tem_palavrao, tox, p1, p2, a1.
  - data/corpus_toxicidade.csv -- a copia de trabalho dos autores, fora do
    controle de versao (.gitignore). Colunas: ID, TITULO, FRASE, PARTE, LINK,
    TOX, P1, P2, A1.
Os rotulos e as anotacoes individuais sao identicos nas duas; o arquivo publicado
tambem ja vem sem as aspas soltas que a planilha de anotacao deixou em tres
frases. load_toxicity_corpus() usa a copia de trabalho quando ela existe e o
arquivo publicado caso contrario, de modo que um clone limpo reproduz o pipeline
inteiro sem nenhum dado a mais.
  - TOX/tox: rótulo consolidado (moda das três anotações, desempate por P1)
  - P1, P2, A1: anotações individuais
Classes: 0=Nenhuma, 1=Leve, 2=Severa

Partição (ver app/nlp/experimento.py, variável YSK_PARTICAO):
  - 'video' (padrão): a divisão é lida de data/particoes/particao_video.csv, em
    que todas as frases de um vídeo ficam do mesmo lado. O vídeo é a unidade em
    que o serviço opera; dividir por frase deixava frases do mesmo vídeo -- mesmos
    falantes, personagens, alvos nomeados e idiossincrasias de transcrição -- em
    treino e em teste ao mesmo tempo.
  - 'frase': a divisão da versão submetida, estratificada por frase. Se o corpus
    contiver frases idênticas, agrupa pelo texto normalizado (StratifiedGroupKFold);
    com todas as frases distintas -- caso do corpus atual -- usa StratifiedKFold.

Rotulagem (YSK_ROTULAGEM): os rótulos são sempre os do corpus v1, de uma só rodada
de anotação; 'v2' (padrão) deixa de fora dos experimentos os grupos de
quase-duplicatas (coluna quase_duplicata) e 'v1' os mantém (ver app/nlp/experimento.py).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
import logging
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp.config import PATHS
from app.nlp import experimento

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rótulos textuais do corpus -> índice de classe
LABEL_MAPPING = {
    'nenhuma': 0,
    'leve': 1,
    'severa': 2,
}

# Nomes alternativos aceitos para as colunas de texto e de rótulo
TEXT_COLUMNS = ('FRASE', 'Frase', 'frase', 'TEXTO', 'Texto')
LABEL_COLUMNS = ('TOX', 'Tox', 'tox')
ANNOTATOR_COLUMNS = ('P1', 'P2', 'A1')
# O arquivo publicado usa video_id onde a copia de trabalho usa ID.
EXTRA_COLUMNS = {'ID': ('ID', 'video_id'), 'PARTE': ('PARTE', 'parte')}

RANDOM_STATE = 42


def _pick_column(df: pd.DataFrame, candidates) -> str:
    """Retorna o primeiro nome de coluna presente no DataFrame."""
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(
        f"Nenhuma das colunas {candidates} encontrada no corpus. "
        f"Colunas disponíveis: {list(df.columns)}"
    )


def _resolve_corpus_path() -> str:
    """
    Escolhe entre a cópia de trabalho e o arquivo publicado.

    A cópia de trabalho está no .gitignore, então num clone limpo do repositório
    público só existe o segundo. Sem este fallback o pipeline inteiro falha ao
    abrir um arquivo que nunca foi distribuído -- o que tornava falsa a promessa
    de reprodução ponta a ponta feita no README e no artigo.
    """
    mestre = PATHS['corpus_toxicidade']
    if os.path.exists(mestre):
        return mestre
    publico = os.path.join(os.path.dirname(mestre), 'corpus_toxicidade_publico.csv')
    if os.path.exists(publico):
        logger.info("Cópia de trabalho ausente; usando o corpus publicado.")
        return publico
    raise FileNotFoundError(
        f"Nenhum corpus encontrado. Procurado em:\n  {mestre}\n  {publico}"
    )


def normalize_text(value) -> str:
    """
    Normaliza uma frase do corpus.

    O CSV guarda as frases entre aspas literais (ex.: '"fodase"'), resquício da
    planilha de anotação. Elas são removidas aqui para que o texto entregue ao
    tokenizador seja o mesmo que o anotador leu.
    """
    text = str(value).strip()
    while len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    return ' '.join(text.split())


def _to_label(value) -> float:
    """Converte um rótulo textual ('Leve') ou numérico ('1') em índice de classe."""
    if pd.isna(value):
        return np.nan
    key = str(value).strip().lower()
    if key in LABEL_MAPPING:
        return LABEL_MAPPING[key]
    try:
        num = int(float(key))
    except ValueError:
        return np.nan
    return num if num in (0, 1, 2) else np.nan


def load_toxicity_corpus(rotulagem: str = None) -> pd.DataFrame:
    """
    Carrega o corpus de toxicidade.

    Args:
        rotulagem: 'v1' (único valor); por padrão, a da rodada (variável
            YSK_ROTULAGEM, ver app/nlp/experimento.py). Conferido aqui para que
            uma rodada com outra rotulagem falhe em vez de usar o v1 calada.

    Returns:
        DataFrame com as colunas 'id_sentenca' (posição 1-based no corpus),
        'texto' (frase normalizada), 'label' (0/1/2), as anotações individuais
        convertidas ('P1', 'P2', 'A1') e, quando existirem, 'ID' (vídeo) e 'PARTE'.
    """
    rotulagem = rotulagem or experimento.rotulagem()
    if rotulagem not in experimento.ROTULAGENS_VALIDAS:
        raise ValueError(f"rotulagem={rotulagem!r}; use um de {experimento.ROTULAGENS_VALIDAS}")
    corpus_path = _resolve_corpus_path()
    logger.info(f"Carregando corpus de toxicidade de: {corpus_path}")

    # sep=None + engine='python' detecta ';' ou ',' automaticamente
    df = pd.read_csv(corpus_path, sep=None, engine='python', encoding='utf-8-sig')
    logger.info(f"Corpus carregado: {len(df)} registros | colunas: {list(df.columns)}")


    text_col = _pick_column(df, TEXT_COLUMNS)
    label_col = _pick_column(df, LABEL_COLUMNS)

    out = pd.DataFrame({
        'id_sentenca': np.arange(1, len(df) + 1),
        'texto': df[text_col].map(normalize_text),
        'label': df[label_col].map(_to_label),
    })

    for col in ANNOTATOR_COLUMNS:
        origem = next((c for c in (col, col.lower()) if c in df.columns), None)
        if origem is not None:
            out[col] = df[origem].map(_to_label)

    for destino, candidatos in EXTRA_COLUMNS.items():
        origem = next((c for c in candidatos if c in df.columns), None)
        if origem is not None:
            out[destino] = df[origem]

    antes = len(out)
    out = out[out['texto'].str.len() > 0]
    out = out.dropna(subset=['label'])
    out['label'] = out['label'].astype(int)

    if len(out) != antes:
        logger.info(f"Removidos {antes - len(out)} registros sem frase ou sem rótulo")

    logger.info(f"Após limpeza: {len(out)} registros")
    logger.info(f"Textos distintos: {out['texto'].str.lower().nunique()}")
    logger.info(f"Distribuição de classes:\n{out['label'].value_counts().sort_index()}")

    return out.reset_index(drop=True)


def _divisao_por_video(df: pd.DataFrame):
    """Índices de treino e teste lidos da partição por vídeo gravada em disco."""
    part = experimento.carregar_particao_video()
    if len(part) != len(df) or not (part['id_sentenca'].values == df['id_sentenca'].values).all():
        raise ValueError("A partição gravada não corresponde ao corpus (id_sentenca). "
                         "Regere com: python -m app.nlp.datasets.gerar_particoes")
    if 'ID' in df.columns and not (part['video_id'].values == df['ID'].astype(str).values).all():
        raise ValueError("video_id da partição não bate com o corpus, linha a linha")
    train_idx = np.where(part['particao'].values == 'treino')[0]
    test_idx = np.where(part['particao'].values == 'teste')[0]
    return train_idx, test_idx, part['video_id'].astype(str).tolist()


def get_data_for_cv_and_test(test_size: float = 0.20, return_groups: bool = False,
                             return_frame: bool = False):
    """
    Prepara dados para Cross-Validation e teste final.

    Com YSK_PARTICAO='video' (padrão), a divisão vem do arquivo de partição por
    vídeo e os grupos devolvidos são os video_id: os folds de CV gerados a partir
    deles (get_kfold_split) também são os gravados no arquivo. `test_size` é então
    ignorado -- a proporção foi fixada ao gerar a partição (20% dos vídeos).

    Com YSK_PARTICAO='frase', reproduz a divisão da versão submetida: agrupada
    pelo texto normalizado, com StratifiedGroupKFold de k = round(1/test_size)
    folds e o primeiro fold como teste (StratifiedKFold quando não há texto
    repetido).

    Args:
        test_size: Proporção aproximada do conjunto de teste (só na partição por frase)
        return_groups: Se True, devolve também os grupos do conjunto de CV,
            necessários para gerar folds sem vazamento.
        return_frame: Se True, devolve por último o DataFrame do corpus e os
            índices (train_idx, test_idx), para quem precisa de id_sentenca/vídeo.

    Returns:
        (X_train_cv, y_train_cv), (X_test, y_test)
        ou, com return_groups=True,
        (X_train_cv, y_train_cv, groups_train_cv), (X_test, y_test)
        e, com return_frame=True, mais (df, train_idx, test_idx) ao final.
    """
    df = load_toxicity_corpus()

    if experimento.particao() == 'video':
        train_idx, test_idx, groups = _divisao_por_video(df)
        texts = df['texto'].tolist()
        labels = df['label'].tolist()
        X_train_cv = [texts[i] for i in train_idx]
        y_train_cv = [labels[i] for i in train_idx]
        g_train_cv = [groups[i] for i in train_idx]
        X_test = [texts[i] for i in test_idx]
        y_test = [labels[i] for i in test_idx]

        compartilhados = set(g_train_cv) & {groups[i] for i in test_idx}
        logger.info(f"=== METODOLOGIA DE DIVISÃO (por vídeo; rotulagem {experimento.rotulagem()}) ===")
        logger.info(f"Total: {len(texts)} frases | treino {len(X_train_cv)} "
                    f"({len(set(g_train_cv))} vídeos) | teste {len(X_test)} "
                    f"({len({groups[i] for i in test_idx})} vídeos)")
        logger.info(f"Vídeos compartilhados entre treino e teste: {len(compartilhados)} (deve ser 0)")
        if compartilhados:
            raise ValueError(f"{len(compartilhados)} vídeos em treino e teste ao mesmo tempo")

        saida = [(X_train_cv, y_train_cv, g_train_cv) if return_groups
                 else (X_train_cv, y_train_cv), (X_test, y_test)]
        if return_frame:
            saida.append((df, train_idx, test_idx))
        return tuple(saida)

    texts = df['texto'].tolist()
    labels = df['label'].tolist()
    groups = df['texto'].str.lower().tolist()

    n_splits = max(2, int(round(1.0 / test_size)))
    ha_repeticao = len(set(groups)) < len(groups)

    if ha_repeticao:
        divisor = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        train_idx, test_idx = next(divisor.split(texts, labels, groups=groups))
    else:
        divisor = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        train_idx, test_idx = next(divisor.split(texts, labels))

    X_train_cv = [texts[i] for i in train_idx]
    y_train_cv = [labels[i] for i in train_idx]
    g_train_cv = [groups[i] for i in train_idx]

    X_test = [texts[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]

    vazamento = set(g_train_cv) & {groups[i] for i in test_idx}

    esquema = 'agrupada por texto' if ha_repeticao else 'estratificada (sem frases repetidas)'
    logger.info(f"=== METODOLOGIA DE DIVISÃO ({esquema}) ===")
    logger.info(f"Total de Amostras: {len(texts)}")
    logger.info(f"Conjunto de TREINO (Para Cross-Validation): {len(X_train_cv)} amostras "
                f"({len(X_train_cv)/len(texts)*100:.1f}%)")
    logger.info(f"Conjunto de TESTE (Reservado): {len(X_test)} amostras "
                f"({len(X_test)/len(texts)*100:.1f}%)")
    logger.info(f"Frases compartilhadas entre treino e teste: {len(vazamento)} (deve ser 0)")

    saida = [(X_train_cv, y_train_cv, g_train_cv) if return_groups
             else (X_train_cv, y_train_cv), (X_test, y_test)]
    if return_frame:
        saida.append((df, train_idx, test_idx))
    return tuple(saida)


if __name__ == "__main__":
    # Teste rápido para verificar carregamento
    (X_train, y_train, g_train), (X_test, y_test) = get_data_for_cv_and_test(
        test_size=0.20, return_groups=True
    )
    print(f"\nDistribuição Treino: {pd.Series(y_train).value_counts().sort_index().to_dict()}")
    print(f"Distribuição Teste: {pd.Series(y_test).value_counts().sort_index().to_dict()}")
    print(f"Grupos distintos no treino: {len(set(g_train))}")
