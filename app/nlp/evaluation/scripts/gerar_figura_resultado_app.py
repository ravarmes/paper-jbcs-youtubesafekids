"""
Gera a Figura 9 do artigo: a lista de resultados da consulta de demonstração com o
score de toxicidade verbal **só do filtro de toxicidade**.

Substitui a captura de tela anterior por dois motivos apontados na revisão:
  * a captura reproduzia miniaturas e logotipos de canais de terceiros (M8c);
  * os selos da captura combinavam três filtros (duração, engajamento e toxicidade), de
    modo que a figura não informava sobre a contribuição do artigo (m10).

A figura é desenhada a partir de results/<rodada>/artigo/scores_exemplo.json, gravado por
calcular_scores_exemplo.py com o modelo de referência da rodada: o título de cada vídeo,
as três frases amostradas (início, meio e fim), a classe predita e a probabilidade de
cada uma, e o score do vídeo (média dos scores das três frases). Nenhum número é digitado
aqui. A ordem é a do protótipo: score decrescente.

Uso (de src/):
    python app/nlp/evaluation/scripts/gerar_figura_resultado_app.py
    python app/nlp/evaluation/scripts/gerar_figura_resultado_app.py --entrada <json> --idioma pt
"""

import argparse
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
)
sys.path.insert(0, project_root)

from app.nlp import experimento  # noqa: E402

NOME = 'fig09-app-resultado'
TEXTOS = {
    'en': {'parte': {'start': 'Beginning', 'middle': 'Middle', 'end': 'End'},
           'classe': {'Nenhuma': 'None', 'Leve': 'Mild', 'Severa': 'Severe'},
           'score': 'Score', 'pos': 'Rank'},
    'pt': {'parte': {'start': 'Início', 'middle': 'Meio', 'end': 'Fim'},
           'classe': {'Nenhuma': 'Nenhuma', 'Leve': 'Leve', 'Severa': 'Severa'},
           'score': 'Score', 'pos': 'Posição'},
}
# Cores do selo por faixa do score (mesma leitura do protótipo: alto = pouco tóxico).
FAIXAS = [(0.80, '#1a7f4b'), (0.50, '#b26b00'), (0.0, '#b3261e')]
ORDEM_PARTES = ['start', 'middle', 'end']


def sem_emoji(s):
    """Remove caracteres fora do plano básico (emoji, bandeiras), que a fonte não desenha."""
    return re.sub(r'[^\u0000-\uffff]', '', s).replace('\u200d', '').strip()


def texto_cabendo(ax, x, y, s, x_max, **kw):
    """Escreve `s` em (x, y) cortando palavras do fim até a largura desenhada caber em x_max."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    palavras = s.split()
    for n in range(len(palavras), 0, -1):
        rotulo = ' '.join(palavras[:n]) + ('' if n == len(palavras) else ' …')
        txt = ax.text(x, y, rotulo, **kw)
        caixa = txt.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
        if caixa.x1 <= x_max:
            return txt
        txt.remove()
    return ax.text(x, y, '…', **kw)


def cor(score):
    for limite, c in FAIXAS:
        if score >= limite:
            return c
    return FAIXAS[-1][1]


def desenhar(dados, idioma, saida_png):
    t = TEXTOS[idioma]
    videos = sorted(dados['videos'].items(),
                    key=lambda kv: kv[1]['score_toxicidade'], reverse=True)

    largura, alt_video = 7.0, 1.42
    fig = plt.figure(figsize=(largura, alt_video * len(videos) + 0.1))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, largura)
    ax.set_ylim(0, alt_video * len(videos) + 0.1)
    ax.axis('off')

    for i, (titulo, v) in enumerate(videos):
        topo = alt_video * (len(videos) - i)
        ax.add_patch(FancyBboxPatch((0.08, topo - alt_video + 0.08), largura - 0.16,
                                    alt_video - 0.14, boxstyle='round,pad=0.02,rounding_size=0.06',
                                    facecolor='#f7f7f7', edgecolor='#c8c8c8', linewidth=0.8))
        ax.text(0.22, topo - 0.22, f'{i + 1}', fontsize=11, fontweight='bold',
                va='center', ha='center', color='#333333')
        texto_cabendo(ax, 0.42, topo - 0.22, sem_emoji(titulo), largura - 1.98,
                      fontsize=8.0, fontweight='bold', va='center', ha='left', color='#222222')

        score = v['score_toxicidade']
        ax.add_patch(FancyBboxPatch((largura - 1.86, topo - 0.36), 1.68, 0.28,
                                    boxstyle='round,pad=0.01,rounding_size=0.08',
                                    facecolor=cor(score), edgecolor='none'))
        ax.text(largura - 1.02, topo - 0.22, f"{t['score']}: {v['score_toxicidade_pct']}%",
                fontsize=8.4, fontweight='bold', color='white', va='center', ha='center')

        frases = {f['parte']: f for f in v['frases']}
        for j, parte in enumerate(ORDEM_PARTES):
            f = frases[parte]
            y = topo - 0.55 - j * 0.30
            classe = f['classe_predita']
            prob = f['probabilidades'][classe]
            ax.text(0.42, y, f"{t['parte'][parte]}:", fontsize=7.4, color='#555555',
                    va='center', ha='left')
            texto_cabendo(ax, 1.12, y, sem_emoji(f['texto']), largura - 1.30,
                          fontsize=7.4, style='italic', color='#222222', va='center', ha='left')
            ax.text(largura - 0.20, y, f"{t['classe'][classe]} ({prob:.2f})", fontsize=7.4,
                    color='#333333', va='center', ha='right')

    fig.savefig(saida_png, dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description='Figura 9: resultado com o score de toxicidade')
    p.add_argument('--entrada', default=None,
                   help='scores_exemplo.json (padrão: o da rodada atual)')
    p.add_argument('--idioma', choices=['en', 'pt'], default='en')
    a = p.parse_args()

    entrada = a.entrada or str(experimento.dir_resultados('artigo') / 'scores_exemplo.json')
    with open(entrada, encoding='utf-8') as fh:
        dados = json.load(fh)
    sufixo = '' if a.idioma == 'en' else '-pt'
    saida = experimento.dir_latex('figuras') / f'{NOME}{sufixo}.png'
    desenhar(dados, a.idioma, saida)
    print(f"modelo: {dados.get('modelo')}")
    print(f'-> {saida}')


if __name__ == '__main__':
    main()
