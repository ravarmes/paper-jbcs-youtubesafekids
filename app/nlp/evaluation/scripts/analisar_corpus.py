"""
Estatísticas descritivas do corpus de toxicidade, para a seção de Metodologia.

Produz, em evaluation/results/corpus/:
  - corpus_stats.json  (todos os números, para conferência)
  - fig-tox-distribuicao-corpus.png (figura do artigo)

Números calculados:
  - tamanho do corpus, vídeos e textos distintos
  - distribuição das três classes
  - concordância entre os três anotadores (unanimidade, acordo par a par,
    kappa de Fleiss)
  - comprimento das sentenças

Uso:
    python -m app.nlp.evaluation.scripts.analisar_corpus
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp.config import PATHS
from app.nlp.datasets.prepare_data_toxicity import load_toxicity_corpus, normalize_text

CLASSES = ['Nenhuma', 'Leve', 'Severa']
CORES = ['#2ecc71', '#f39c12', '#e74c3c']
ANOTADORES = ['P1', 'P2', 'A1']

# Termos de baixo calão usados no diagnóstico de decidibilidade lexical. Não é
# um léxico de toxicidade — serve só para medir o quanto o rótulo do corpus
# pode ser adivinhado pela presença de um palavrão.
PALAVROES = (
    r'\b(merda|porra|caralho|foda|fodase|bosta|puta|cu|buceta|viado|idiota|'
    r'imbecil|babaca|vadia|piranha|desgraçad\w*|lixo|burr\w+|otari\w+|corno|'
    r'arrombad\w+|cacete|inferno|droga)\w*\b'
)


def fleiss_kappa(matriz_contagem: np.ndarray) -> float:
    """
    Kappa de Fleiss para múltiplos anotadores.

    Args:
        matriz_contagem: matriz (n_itens x n_categorias) em que cada célula é
            o número de anotadores que atribuíram aquela categoria ao item.

    Returns:
        Valor de kappa.
    """
    n_itens, _ = matriz_contagem.shape
    n_anotadores = matriz_contagem.sum(axis=1)[0]

    # Concordância observada por item
    p_i = (np.sum(matriz_contagem ** 2, axis=1) - n_anotadores) / (
        n_anotadores * (n_anotadores - 1)
    )
    p_bar = p_i.mean()

    # Concordância esperada ao acaso
    p_j = matriz_contagem.sum(axis=0) / (n_itens * n_anotadores)
    p_e = np.sum(p_j ** 2)

    return (p_bar - p_e) / (1 - p_e)


def concordancia(df: pd.DataFrame) -> dict:
    """Calcula as métricas de concordância entre os anotadores disponíveis."""
    presentes = [c for c in ANOTADORES if c in df.columns]
    if len(presentes) < 2:
        return {'anotadores_disponiveis': presentes}

    anot = df[presentes].astype(int)

    unanimes = (anot.nunique(axis=1) == 1).sum()
    maioria = (anot.apply(lambda r: r.value_counts().iloc[0], axis=1) >= 2).sum()

    pares = {}
    for a, b in combinations(presentes, 2):
        pares[f'{a}-{b}'] = float((anot[a] == anot[b]).mean())

    contagem = np.zeros((len(anot), len(CLASSES)), dtype=int)
    for idx, (_, linha) in enumerate(anot.iterrows()):
        for valor in linha:
            contagem[idx, int(valor)] += 1

    return {
        'anotadores_disponiveis': presentes,
        'n_anotadores': len(presentes),
        'unanimidade_n': int(unanimes),
        'unanimidade_pct': float(unanimes / len(anot)),
        'com_maioria_n': int(maioria),
        'sem_maioria_n': int(len(anot) - maioria),
        'acordo_par_a_par': pares,
        'acordo_par_a_par_medio': float(np.mean(list(pares.values()))),
        'fleiss_kappa': float(fleiss_kappa(contagem)),
    }


def decidibilidade_lexical(textos, rotulos) -> dict:
    """
    Mede o quanto o rótulo pode ser adivinhado pelo vocabulário isolado.

    Um corpus em que a classe é recuperável de uma única palavra mede
    memorização de léxico ofensivo, não detecção de toxicidade: modelos de
    sacola de palavras operam perto do teto e o contexto deixa de informar.
    Dois indicadores, ambos independentes de qualquer modelo treinado:

    1. fração de sentenças que contêm alguma palavra associada exclusivamente
       a uma das classes (entre as de frequência mínima);
    2. desempenho da regra ingênua "tem palavrão, logo é tóxica", que serve de
       piso trivial para a tarefa binária tóxico/não-tóxico.
    """
    from sklearn.metrics import f1_score

    ocorrencias = defaultdict(Counter)
    for texto, rotulo in zip(textos, rotulos):
        for palavra in set(re.findall(r'\w+', texto.lower())):
            ocorrencias[palavra][rotulo] += 1

    puras = {
        palavra for palavra, contagem in ocorrencias.items()
        if sum(contagem.values()) >= 5 and max(contagem.values()) == sum(contagem.values())
    }
    cobertas = sum(
        1 for t in textos if any(w in puras for w in re.findall(r'\w+', t.lower()))
    )

    tem_palavrao = [bool(re.search(PALAVROES, t.lower())) for t in textos]
    predito = [1 if p else 0 for p in tem_palavrao]
    real = [1 if r > 0 else 0 for r in rotulos]

    n_nao_toxicas = sum(1 for r in rotulos if r == 0)
    n_toxicas = len(rotulos) - n_nao_toxicas
    nao_toxica_com = sum(1 for p, r in zip(tem_palavrao, rotulos) if p and r == 0)
    toxica_sem = sum(1 for p, r in zip(tem_palavrao, rotulos) if not p and r > 0)

    return {
        'palavras_de_classe_pura': len(puras),
        'sentencas_decidiveis_por_palavra': cobertas,
        'pct_decidiveis_por_palavra': round(cobertas / len(textos) * 100, 1),
        'sentencas_com_palavrao': sum(tem_palavrao),
        'pct_com_palavrao': round(sum(tem_palavrao) / len(textos) * 100, 1),
        'regra_ingenua_palavrao_f1_macro': round(
            float(f1_score(real, predito, average='macro')), 4
        ),
        'nao_toxicas_com_palavrao': nao_toxica_com,
        'pct_nao_toxicas_com_palavrao': round(nao_toxica_com / max(n_nao_toxicas, 1) * 100, 1),
        'toxicas_sem_palavrao': toxica_sem,
        'pct_toxicas_sem_palavrao': round(toxica_sem / max(n_toxicas, 1) * 100, 1),
    }


def proximidade_treino_teste(test_size=0.20, limiar=0.8) -> dict:
    """
    Mede quantas frases do teste têm um análogo próximo no treino.

    Sem repetições exatas ainda pode haver quase-duplicatas — frases que
    diferem por uma ou duas palavras. Modelos lexicais acertam essas quase por
    construção, então a fração delas indica o quanto a métrica agregada mede
    generalização de fato.
    """
    from app.nlp.datasets.prepare_data_toxicity import get_data_for_cv_and_test
    from app.nlp.evaluation.evaluate_toxicity import separar_por_similaridade

    (treino, _), (teste, _) = get_data_for_cv_and_test(test_size=test_size)
    com, sem, _ = separar_por_similaridade(treino, teste, limiar=limiar)

    return {
        'limiar': limiar,
        'n_teste': len(teste),
        'com_analogo_proximo': len(com),
        'sem_analogo_proximo': len(sem),
        'pct_com_analogo_proximo': round(len(com) / len(teste) * 100, 1),
    }


def figura_distribuicao(distribuicao, output_path):
    """Gera a figura de distribuição das classes usada no artigo."""
    total = sum(distribuicao)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    barras = ax.bar(CLASSES, distribuicao, color=CORES, edgecolor='white', linewidth=1.2)
    for barra, valor in zip(barras, distribuicao):
        ax.annotate(
            f'{valor}\n({valor/total*100:.1f}%)',
            xy=(barra.get_x() + barra.get_width() / 2, valor),
            xytext=(0, 4), textcoords='offset points',
            ha='center', va='bottom', fontsize=10
        )

    ax.set_ylabel('Sentenças anotadas', fontsize=11)
    ax.set_xlabel('Grau de toxicidade', fontsize=11)
    ax.set_ylim(0, max(distribuicao) * 1.18)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  figura: {output_path}")


def main():
    df = load_toxicity_corpus()

    # Vídeos: recuperados do CSV bruto (o loader não propaga LINK/TITULO)
    bruto = pd.read_csv(
        PATHS['corpus_toxicidade'], sep=None, engine='python', encoding='utf-8-sig'
    )
    n_videos = bruto['ID'].nunique() if 'ID' in bruto.columns else None

    texto_norm = df['texto'].str.lower()
    distribuicao = [int((df['label'] == i).sum()) for i in range(len(CLASSES))]
    palavras = df['texto'].str.split().str.len()

    repetidos = texto_norm.value_counts()
    repetidos = repetidos[repetidos > 1]

    stats = {
        'total_sentencas': int(len(df)),
        'textos_distintos': int(texto_norm.nunique()),
        'n_videos': int(n_videos) if n_videos is not None else None,
        'distribuicao': dict(zip(CLASSES, distribuicao)),
        'distribuicao_pct': {
            c: round(v / len(df) * 100, 1) for c, v in zip(CLASSES, distribuicao)
        },
        'textos_repetidos': {
            'n_textos': int(len(repetidos)),
            'n_linhas': int(repetidos.sum()),
            'pct_linhas': round(float(repetidos.sum() / len(df)) * 100, 1),
        },
        'comprimento_palavras': {
            'media': round(float(palavras.mean()), 1),
            'mediana': float(palavras.median()),
            'p95': float(palavras.quantile(0.95)),
            'max': int(palavras.max()),
        },
        'concordancia': concordancia(df),
        'decidibilidade_lexical': decidibilidade_lexical(
            df['texto'].tolist(), df['label'].tolist()
        ),
        'proximidade_treino_teste': proximidade_treino_teste(),
    }

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'corpus'
    )
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'corpus_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    figura_distribuicao(
        distribuicao, os.path.join(output_dir, 'fig02-distribuicao-corpus.png')
    )

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nEstatísticas salvas em: {output_dir}")


if __name__ == '__main__':
    main()
