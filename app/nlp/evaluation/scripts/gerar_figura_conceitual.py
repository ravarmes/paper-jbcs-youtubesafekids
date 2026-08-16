"""
Gera a figura conceitual do artigo: o pipeline de ponta a ponta, do corpus ao
servico que reordena a busca.

Existe porque o JISA encoraja "at least one color conceptual figure representing
the key research questions or solutions", e porque a figura de fluxo herdada da
pasta da conferencia (fig-tox-fluxometodologico.png) esta em portugues e
desatualizada -- fala em 2.998 sentencas, Focal Loss e 2 estagios, numeros da
versao anterior do trabalho.

Todo numero exibido e lido de resumo_numeros.json: a figura nao pode divergir do
texto porque nao ha valor digitado a mao nela.

Uso:
    python -m app.nlp.evaluation.scripts.gerar_figura_conceitual --idioma en
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'
)
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
)
FIGURAS_DIR = Path(project_root).parent / 'latex' / 'figuras'

COR_TITULO = '#1e293b'
COR_BORDA = '#94a3b8'
CORES_ETAPA = ['#2980b9', '#16a085', '#8e44ad', '#c0392b', '#d35400']
PAD_BOX = 0.010


def _num(valor, casas=4, idioma='en'):
    texto = f'{valor:.{casas}f}'
    return texto if idioma == 'en' else texto.replace('.', ',')


def _mil(n, idioma='en'):
    return f'{n:,}' if idioma == 'en' else f'{n:,}'.replace(',', '.')


def _caixa(ax, x, y, w, h, titulo, linhas, cor):
    """Desenha uma caixa com faixa de titulo e linhas de corpo."""
    # Sombra sutil
    ax.add_patch(FancyBboxPatch(
        (x + 0.002, y - 0.003), w, h, boxstyle=f'round,pad={PAD_BOX},rounding_size=0.02',
        linewidth=0, facecolor='#000000', alpha=0.06, zorder=1))
    # Caixa principal
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f'round,pad={PAD_BOX},rounding_size=0.02',
        linewidth=1.2, edgecolor=COR_BORDA, facecolor='#f8fafc', zorder=2))
    # Faixa de titulo
    faixa_h = 0.092
    faixa_margem = 0.008
    ax.add_patch(FancyBboxPatch(
        (x + faixa_margem, y + h - faixa_h - faixa_margem),
        w - 2 * faixa_margem, faixa_h,
        boxstyle='round,pad=0.004,rounding_size=0.012',
        linewidth=0, facecolor=cor, zorder=3))
    ax.text(x + w / 2, y + h - faixa_margem - faixa_h / 2, titulo, ha='center',
            va='center', fontsize=9.5, fontweight='bold', color='white', zorder=4)
    # Linhas de texto
    n_linhas = len(linhas)
    topo_texto = y + h - faixa_h - faixa_margem - 0.038
    espacamento = 0.048
    for i, linha in enumerate(linhas):
        ax.text(x + w / 2, topo_texto - i * espacamento, linha,
                ha='center', va='top', fontsize=8.2, color=COR_TITULO, zorder=4)


def _seta(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle='-|>', mutation_scale=13, linewidth=1.4,
        color=COR_TITULO, shrinkA=0, shrinkB=0, zorder=5))


def gerar(numeros, idioma='en'):
    c = numeros['corpus']
    lex = c['decidibilidade_lexical']
    split = numeros['split']
    cv = numeros['bert_cv']
    teste = numeros['bert_teste']
    prox = c['proximidade_treino_teste']

    if idioma == 'en':
        etapas = [
            ('1. Corpus construction', [
                f"{_mil(c['total_sentencas'])} sentences from {c['n_videos']} videos",
                f"three raters, Fleiss' $\\kappa$ = "
                f"{_num(c['concordancia']['fleiss_kappa'], 4)}",
                f"{lex['pct_toxicas_sem_palavrao']}% of toxic sentences",
                'carry no profanity at all',
            ]),
            ('2. Stratified split', [
                f"{_mil(split['treino_cv'])} for 5-fold cross-validation",
                f"{split['teste']} held out, never consulted",
                f"{prox['sem_analogo_proximo']} test sentences have no",
                'near analogue in training',
            ]),
            ('3. Hyperparameter search', [
                'BERTimbau, 9 search dimensions',
                'random search + Successive Halving',
                '32 $\\rightarrow$ 10 $\\rightarrow$ 3 configurations',
                '159 training runs, 3 seeds',
            ]),
            ('4a. Offline evaluation', [
                f"macro F1 {_num(cv['f1_macro'])} (cross-val.)",
                f"macro F1 {_num(teste['f1_macro'])} (held-out test)",
                'vs. TF-IDF baselines, a few-shot LLM',
                'and off-the-shelf classifiers',
            ]),
            ('4b. Reranking service', [
                'safety score from the full',
                'probability distribution',
                'reorders YouTube search results',
                'operating point set by error cost',
            ]),
        ]
        rodape = ('Research question: can a toxicity signal learned natively in Brazilian Portuguese '
                  'reorder children’s video search results by a score a caregiver can inspect?')
    else:
        etapas = [
            ('1. Construção do corpus', [
                f"{_mil(c['total_sentencas'], 'pt')} sentenças de {c['n_videos']} vídeos",
                'três avaliadores, $\\kappa$ de Fleiss = '
                f"{_num(c['concordancia']['fleiss_kappa'], 4, 'pt')}",
                f"{lex['pct_toxicas_sem_palavrao']}% das tóxicas".replace('.', ','),
                'não contêm palavrão algum',
            ]),
            ('2. Divisão estratificada', [
                f"{_mil(split['treino_cv'], 'pt')} para validação cruzada",
                f"{split['teste']} retidos, jamais consultados",
                f"{prox['sem_analogo_proximo']} sentenças de teste sem",
                'análogo próximo no treino',
            ]),
            ('3. Busca de hiperparâmetros', [
                'BERTimbau, 9 dimensões',
                'busca aleatória + Successive Halving',
                '32 $\\rightarrow$ 10 $\\rightarrow$ 3 configurações',
                '159 treinamentos, 3 sementes',
            ]),
            ('4a. Avaliação offline', [
                f"F1 macro {_num(cv['f1_macro'], 4, 'pt')} (val. cruzada)",
                f"F1 macro {_num(teste['f1_macro'], 4, 'pt')} (teste retido)",
                'contra baselines TF-IDF, LLM few-shot',
                'e classificadores prontos',
            ]),
            ('4b. Serviço de reordenação', [
                'score de segurança a partir de toda',
                'a distribuição de probabilidade',
                'reordena resultados do YouTube',
                'ponto de operação pelo custo do erro',
            ]),
        ]
        rodape = ('Pergunta de pesquisa: um sinal de toxicidade aprendido nativamente em português '
                  'pode reordenar buscas de vídeo infantil por um score auditável pelo responsável?')

    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    larg, alt = 0.280, 0.365
    y_topo, y_base = 0.560, 0.080
    xs_topo = [0.030, 0.360, 0.690]

    for i in range(3):
        _caixa(ax, xs_topo[i], y_topo, larg, alt, etapas[i][0], etapas[i][1],
               CORES_ETAPA[i])

    for i in range(2):
        _seta(ax, (xs_topo[i] + larg + PAD_BOX, y_topo + alt / 2),
              (xs_topo[i + 1] - PAD_BOX, y_topo + alt / 2))

    xs_base = [0.185, 0.535]
    for k, i in enumerate((3, 4)):
        _caixa(ax, xs_base[k], y_base, larg, alt, etapas[i][0], etapas[i][1],
               CORES_ETAPA[i])

    # Da caixa 3 descem duas setas, uma para cada saida (4a e 4b)
    x_saida = xs_topo[2] + larg / 2
    y_junta = 0.505
    y_topo_bottom = y_topo - PAD_BOX
    y_base_top = y_base + alt + PAD_BOX

    # Linha vertical saindo da base da caixa 3
    ax.plot([x_saida, x_saida], [y_topo_bottom, y_junta], color=COR_TITULO,
            linewidth=1.4, zorder=5)
    # Linha horizontal conectando o ramo ate a caixa 4a
    ax.plot([xs_base[0] + larg / 2, x_saida], [y_junta, y_junta], color=COR_TITULO,
            linewidth=1.4, zorder=5)

    # Setas verticais descendo para 4a e 4b
    for x in (xs_base[0] + larg / 2, xs_base[1] + larg / 2):
        _seta(ax, (x, y_junta), (x, y_base_top))

    ax.text(0.5, 0.020, rodape, ha='center', va='bottom', fontsize=8.5,
            style='italic', color=COR_TITULO, wrap=True)

    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    nome_arquivo = 'fig01-pipeline.png' if idioma == 'en' else 'fig01-pipeline-pt.png'
    caminho = FIGURAS_DIR / nome_arquivo
    fig.savefig(caminho, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return caminho


def main():
    parser = argparse.ArgumentParser(description='Gera a figura conceitual do pipeline')
    parser.add_argument('--idioma', choices=['pt', 'en'], default='en')
    args = parser.parse_args()

    caminho_json = os.path.join(RESULTS_DIR, 'artigo', 'resumo_numeros.json')
    with open(caminho_json, 'r', encoding='utf-8') as f:
        numeros = json.load(f)

    print(f'Números de: {os.path.basename(caminho_json)}')
    print(f'  {gerar(numeros, args.idioma)}')


if __name__ == '__main__':
    main()
