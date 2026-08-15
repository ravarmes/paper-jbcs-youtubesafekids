"""
Regera as saidas LaTeX do estudo de caso a partir dos JSON de resultado ja
salvos, sem precisar do modelo treinado.

Existe porque `analisar_estratos.py` e `simular_amostragem.py` carregam os pesos
do BERTimbau para recalcular tudo, e os pesos ficam fora do git. Para trocar o
idioma das saidas do artigo do JIS basta reformatar numeros que ja foram
computados — nenhum valor e digitado a mao.

Entradas (as mais recentes de evaluation/results/):
  estratos_*.json
  simulacao_amostragem_*.json

Saidas:
  latex/tabelas/tab-estratos.tex
  latex/tabelas/tab-simulacao.tex
  latex/figuras/fig-tox-estratos.png

Uso:
    python -m app.nlp.evaluation.scripts.regerar_tabelas_estudo_caso --idioma en
"""

import argparse
import glob
import json
import os
from pathlib import Path

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'
)
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
)
TABELAS_DIR = Path(project_root).parent / 'latex' / 'tabelas'
FIGURAS_DIR = Path(project_root).parent / 'latex' / 'figuras'

# Rotulo de cada regra de agregacao, na ordem em que aparece na tabela.
AGREGACOES = [
    ('media', 'Média das $k$ (Eq.\\ atual)', 'Mean of the $k$ (current Eq.)'),
    ('minimo', 'Mínimo das $k$', 'Minimum of the $k$'),
    ('min_ponderado',
     '\\textbf{$0{,}7\\cdot$mín $+\\ 0{,}3\\cdot$média}',
     '\\textbf{$0.7\\cdot$min $+\\ 0.3\\cdot$mean}'),
    ('duas_piores', 'Média das 2 piores', 'Mean of the 2 lowest'),
]

KS = ['3', '5', '8', '12']

# O template do JIS e de duas colunas: com --duas-colunas as tabelas saem como
# table*, que ocupa a largura da pagina.
DUAS_COLUNAS = False


def _abre_tabela():
    return '\\begin{table*}[!t]' if DUAS_COLUNAS else '\\begin{table}[!t]'


def _fecha_tabela():
    return '\\end{table*}' if DUAS_COLUNAS else '\\end{table}'


def _mais_recente(padrao):
    arquivos = glob.glob(os.path.join(RESULTS_DIR, padrao))
    if not arquivos:
        raise FileNotFoundError(f'Nenhum arquivo casa com {padrao} em {RESULTS_DIR}')
    return max(arquivos, key=os.path.getmtime)


def _carregar(padrao):
    caminho = _mais_recente(padrao)
    with open(caminho, 'r', encoding='utf-8') as f:
        return json.load(f), os.path.basename(caminho)


def _dec(texto, idioma):
    return texto if idioma == 'en' else texto.replace('.', '{,}')


def tabela_estratos(dados, idioma):
    if idioma == 'en':
        legenda = (
            '\\caption{Distribution of the safety score across the four queries of the '
            'case study, ten videos each. In stratum~A the desirable outcome is that '
            'scores be high and close to one another; in stratum~B, that they be low. '
            'Queries are reproduced verbatim in Brazilian Portuguese because they are '
            'inputs to the system; English translations are given in the text.}')
        cabecalho = 'Stratum & Query & $n$ & Mean & Minimum & Range \\\\'
        agregado = 'aggregate'
    else:
        legenda = (
            '\\caption{Distribuição do \\emph{score} de segurança nas quatro buscas do '
            'estudo de caso, com dez vídeos cada. No estrato A o desejável é que os '
            '\\emph{scores} sejam altos e próximos entre si; no B, que sejam baixos.}')
        cabecalho = 'Estrato & Termo de busca & $n$ & Média & Mínimo & Amplitude \\\\'
        agregado = 'agregado'

    nomes = {'A_responsavel': 'A', 'B_apelo_infantil': 'B'}
    linhas = [_abre_tabela(), '\\centering', legenda,
              '\\label{tab:estratos}',
              '\\begin{tabular}{llcccc}', '\\hline', cabecalho, '\\hline']

    for termo, d in dados['por_termo'].items():
        seguro = termo.replace('&', '\\&')
        linhas.append(_dec(
            f"{nomes[d['estrato']]} & \\emph{{{seguro}}} & {d['n']} & "
            f"{d['media']:.3f} & {d['min']:.3f} & {d['amplitude']:.3f} \\\\", idioma))

    linhas.append('\\hline')
    for chave, rot in (('estrato_A_responsavel', 'A'), ('estrato_B_apelo_infantil', 'B')):
        d = dados[chave]
        linhas.append(_dec(
            f"\\textbf{{{rot}}} & \\emph{{{agregado}}} & {d['n']} & "
            f"\\textbf{{{d['media']:.3f}}} & {d['min']:.3f} & {d['amplitude']:.3f} \\\\",
            idioma))

    linhas += ['\\hline', '\\end{tabular}', _fecha_tabela(), '']
    return '\n'.join(linhas)


def tabela_simulacao(dados, idioma, f='0.20'):
    n_sim = dados['n_simulacoes']
    alarme = dados['falso_alarme_alvo_pct']
    pct_toxico = int(float(f) * 100)

    # O separador de milhar do ingles e a virgula, que o LaTeX espaca como
    # pontuacao dentro de formula; fora de $...$ nao ha problema, mas o numero
    # e montado a parte para nao mexer nas virgulas do resto da legenda.
    n_sim_txt = f'{n_sim:,}'

    if idioma == 'en':
        legenda = (
            f'\\caption{{Simulated detection of a video containing {pct_toxico}\\% severely '
            f'toxic content, as a function of the number $k$ of sampled sentences and of '
            f'the aggregation rule. Each rule is calibrated to a {alarme}\\% false alarm '
            f'rate on entirely appropriate videos, so that the columns are comparable. '
            f'Sampling population: the {dados.get("n_populacao", 302)} sentences of the '
            f'held-out test set; {n_sim_txt} simulated videos per cell.}}')
        cabecalho = 'Aggregation & $k=3$ & $k=5$ & $k=8$ & $k=12$ \\\\'
    else:
        legenda = (
            f'\\caption{{Detecção simulada de vídeo com {pct_toxico}\\% de conteúdo '
            f'severamente tóxico, em função do número $k$ de sentenças amostradas e da '
            f'regra de agregação. Cada regra é calibrada para {alarme}\\% de falso alarme '
            f'sobre vídeos inteiramente apropriados, de modo que as colunas sejam '
            f'comparáveis. População de amostragem: as {dados.get("n_populacao", 302)} '
            f'sentenças do teste retido.}}')
        cabecalho = 'Agregação & $k=3$ & $k=5$ & $k=8$ & $k=12$ \\\\'

    linhas = [_abre_tabela(), '\\centering', legenda,
              '\\label{tab:simulacao}',
              '\\begin{tabular}{lcccc}', '\\hline', cabecalho, '\\hline']

    bloco = dados['deteccao'][f]
    for chave, rot_pt, rot_en in AGREGACOES:
        rotulo = rot_en if idioma == 'en' else rot_pt
        celulas = [f"{round(bloco[chave][k]['deteccao'] * 100)}\\%" for k in KS]
        if chave == 'min_ponderado':
            celulas = [f'\\textbf{{{c}}}' for c in celulas]
        linhas.append(' & '.join([rotulo] + celulas) + ' \\\\')

    linhas += ['\\hline', '\\end{tabular}', _fecha_tabela(), '']
    return '\n'.join(linhas)


def figura_estratos(dados, idioma):
    """Dispersao dos scores por termo, reconstruida de `deslocamentos`.

    Reproduz a figura de `analisar_estratos.py` sem carregar o modelo: os 40
    scores por video ja estao salvos no JSON.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    COR_A, COR_B = '#2ecc71', '#e74c3c'
    limiar = dados['falhas_estrato_b']['limiar']
    rng = np.random.default_rng(3)

    # Ordem dos termos: a mesma de por_termo, que ja vem agrupada por estrato.
    termos = list(dados['por_termo'].keys())
    por_termo = {t: [] for t in termos}
    for d in dados['deslocamentos']:
        por_termo[d['termo']].append(d['score'])

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    rotulos = []
    for i, termo in enumerate(termos):
        estrato = dados['por_termo'][termo]['estrato']
        cor = COR_A if estrato == 'A_responsavel' else COR_B
        s = por_termo[termo]
        x = i + rng.uniform(-0.16, 0.16, len(s))
        ax.scatter(x, s, s=52, color=cor, alpha=0.85, edgecolor='white',
                   linewidth=1.1, zorder=3)
        ax.plot([i - 0.30, i + 0.30], [np.mean(s)] * 2, color='#2c3e50',
                linewidth=1.8, zorder=4)
        rotulos.append(termo if len(termo) <= 22 else termo[:20] + '…')

    if idioma == 'en':
        texto_limiar = f'threshold {limiar:.2f}'
        rot_a, rot_b = 'A — caregiver query', 'B — child appeal'
        rot_y = 'Video safety score'
    else:
        texto_limiar = f'limiar {limiar:.2f}'.replace('.', ',')
        rot_a, rot_b = 'A — busca do responsável', 'B — apelo infantil'
        rot_y = 'Score de segurança do vídeo'

    ax.axhline(limiar, color='#7f8c8d', linestyle='--', linewidth=1.3, zorder=2)
    ax.text(len(termos) - 0.52, limiar + 0.012, texto_limiar, fontsize=9, color='#7f8c8d')
    ax.axvline(1.5, color='#bdc3c7', linewidth=1.0, zorder=1)
    ax.text(0.5, 0.055, rot_a, ha='center', fontsize=9.5, color=COR_A, fontweight='bold')
    ax.text(2.5, 0.055, rot_b, ha='center', fontsize=9.5, color=COR_B, fontweight='bold')

    ax.set_xticks(range(len(termos)))
    ax.set_xticklabels(rotulos, fontsize=9)
    ax.set_ylabel(rot_y, fontsize=10)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.5, len(termos) - 0.5)
    ax.grid(axis='y', alpha=0.25, linestyle=':')
    ax.set_axisbelow(True)
    for lado in ('top', 'right'):
        ax.spines[lado].set_visible(False)

    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    caminho = FIGURAS_DIR / 'fig-tox-estratos.png'
    fig.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return caminho


def main():
    global DUAS_COLUNAS

    parser = argparse.ArgumentParser(
        description='Regera as saídas do estudo de caso a partir dos JSON salvos')
    parser.add_argument('--idioma', choices=['pt', 'en'], default='pt')
    parser.add_argument('--fracao-toxica', default='0.20',
                        help='Chave da fração de conteúdo tóxico simulada (padrão 0.20)')
    parser.add_argument('--duas-colunas', action='store_true',
                        help='Emite table* em vez de table, para o template do JIS')
    args = parser.parse_args()

    DUAS_COLUNAS = args.duas_colunas

    TABELAS_DIR.mkdir(parents=True, exist_ok=True)

    estratos, arq_e = _carregar('estratos_*.json')
    simulacao, arq_s = _carregar('simulacao_amostragem_*.json')
    print(f'Fontes:\n  estratos:  {arq_e}\n  simulação: {arq_s}\n')

    for nome, conteudo in (
        ('tab-estratos.tex', tabela_estratos(estratos, args.idioma)),
        ('tab-simulacao.tex', tabela_simulacao(simulacao, args.idioma, args.fracao_toxica)),
    ):
        destino = TABELAS_DIR / nome
        destino.write_text(conteudo, encoding='utf-8')
        print(f'  {destino}')

    print(f'  {figura_estratos(estratos, args.idioma)}')


if __name__ == '__main__':
    main()
