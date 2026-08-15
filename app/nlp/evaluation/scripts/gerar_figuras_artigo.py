"""
Gera as figuras e o trecho .tex da seção de Resultados a partir dos arquivos
produzidos pelo treinamento — sem nenhum número digitado à mão.

Entradas (as mais recentes de cada pasta):
  evaluation/results/toxicity_gridsearch/gridsearch_optimized_*.json
  evaluation/results/toxicity_evaluation/evaluation_metrics_TOX_*.json
  evaluation/results/baselines_toxicity_*.json
  evaluation/results/corpus/corpus_stats.json

Saídas (em evaluation/results/artigo/):
  fig-tox-distribuicao-corpus.png
  fig-tox-comparativo-f1.png
  fig-tox-matriz-confusao.png
  fig-tox-metricas-classe.png
  fig-tox-f1-por-classe-comparativo.png
  resultados.tex        — tabelas prontas para \\input no main.tex
  resumo_numeros.json   — todo número citado no artigo, para conferência

Uso:
    python -m app.nlp.evaluation.scripts.gerar_figuras_artigo
    python -m app.nlp.evaluation.scripts.gerar_figuras_artigo --copiar-para ../latex/figuras

Idioma: o artigo do JIS e em ingles, entao rotulos de eixo, nomes de classe,
separador decimal e texto das tabelas mudam com --idioma en. As chaves internas
dos JSON continuam em portugues (sao dados, nao texto do artigo).

    python -m app.nlp.evaluation.scripts.gerar_figuras_artigo --idioma en \
        --copiar-para ../latex/figuras
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

CLASSES = ['Nenhuma', 'Leve', 'Severa']
CORES = ['#2ecc71', '#f39c12', '#e74c3c']
COR_BERT = '#2980b9'
COR_BASE = '#95a5a6'

# --- idioma de saida -------------------------------------------------------
# CLASSES continua em portugues porque e chave de dicionario nos JSON de
# entrada; o que muda com o idioma e so o texto exibido.
IDIOMA = 'pt'

TRADUCOES = {
    # classes
    'Nenhuma': 'None',
    'Leve': 'Mild',
    'Severa': 'Severe',
    # eixos e legendas
    'Sentenças anotadas': 'Annotated sentences',
    'Grau de toxicidade': 'Toxicity level',
    'F1 (macro)': 'F1 (macro)',
    'Validação cruzada': 'Cross-validation',
    'Teste retido': 'Held-out test',
    'Teste, sem análogo próximo': 'Held-out test, no near analogue',
    'Classe predita': 'Predicted class',
    'Classe real': 'True class',
    'Precisão': 'Precision',
    'Recall': 'Recall',
    'F1': 'F1',
    'Score': 'Score',
    'F1 por classe': 'Per-class F1',
    'Classe anotada': 'Annotated class',
    'Score de segurança': 'Safety score',
    'F1 macro no teste retido': 'Macro F1 on the held-out test set',
    'mediana': 'median',
    # nomes de modelos que vem dos JSON de resultado
    'BERTimbau (ajuste fino)': 'BERTimbau (fine-tuned)',
    'Zero-shot por máscara (melhor molde)': 'Masked zero-shot (best template)',
    'Modelos prontos compostos (HateBR + alvo)': 'Off-the-shelf models composed (HateBR + target)',
    'HateBR isolado (tóxico vs. não-tóxico)': 'HateBR alone (toxic vs. non-toxic)',
    'Classificador de alvo isolado (Leve vs. Severa)': 'Target classifier alone (Mild vs. Severe)',
}


def _t(texto):
    """Traduz um rotulo de exibicao; no modo pt devolve o proprio texto."""
    if IDIOMA != 'en':
        return texto
    return TRADUCOES.get(texto, texto)


def _classes_exibicao():
    return [_t(c) for c in CLASSES]


def _sem_latex(texto):
    """Remove marcacao LaTeX de um rotulo destinado ao matplotlib."""
    return re.sub(r'\\emph\{([^}]*)\}', r'\1', texto)


# O template do JIS e de duas colunas e estas tabelas nao cabem na largura de
# uma coluna: com --duas-colunas elas saem como table*, que ocupa a pagina.
DUAS_COLUNAS = False


def _abre_tabela():
    return '\\begin{table*}[!t]' if DUAS_COLUNAS else '\\begin{table}[!t]'


def _fecha_tabela():
    return '\\end{table*}' if DUAS_COLUNAS else '\\end{table}'

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'
)
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'artigo')


def _mais_recente(padrao):
    """Retorna o arquivo mais recente que casa com o padrão, ou None."""
    arquivos = glob.glob(padrao)
    return max(arquivos, key=os.path.getmtime) if arquivos else None


def _carregar(padrao, obrigatorio=True):
    caminho = _mais_recente(padrao)
    if caminho is None:
        if obrigatorio:
            raise FileNotFoundError(
                f"Nenhum arquivo encontrado para {padrao}. "
                f"Rode o treinamento/avaliação antes de gerar as figuras."
            )
        return None, None
    with open(caminho, 'r', encoding='utf-8') as f:
        return json.load(f), os.path.basename(caminho)


def _virgula(valor, casas=3):
    """Formata número com o separador decimal do idioma do artigo."""
    texto = f'{valor:.{casas}f}'
    return texto if IDIOMA == 'en' else texto.replace('.', ',')


def _finalizar(fig, ax_list, nome):
    for ax in np.atleast_1d(ax_list).ravel():
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True)
    caminho = os.path.join(OUTPUT_DIR, nome)
    fig.tight_layout()
    fig.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  {nome}")
    return caminho


# ----------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------

def fig_distribuicao_corpus(corpus):
    distribuicao = [corpus['distribuicao'][c] for c in CLASSES]
    total = sum(distribuicao)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    barras = ax.bar(_classes_exibicao(), distribuicao, color=CORES,
                    edgecolor='white', linewidth=1.2)
    for barra, valor in zip(barras, distribuicao):
        ax.annotate(f'{valor}\n({_virgula(valor / total * 100, 1)}%)',
                    xy=(barra.get_x() + barra.get_width() / 2, valor),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10)
    ax.set_ylabel(_t('Sentenças anotadas'), fontsize=11)
    ax.set_xlabel(_t('Grau de toxicidade'), fontsize=11)
    ax.set_ylim(0, max(distribuicao) * 1.20)
    ax.grid(axis='y', alpha=0.25)
    return _finalizar(fig, ax, 'fig-tox-distribuicao-corpus.png')


def fig_comparativo_f1(modelos):
    """
    Barras agrupadas: F1 macro na validação cruzada, no teste retido e no
    teste contado por texto distinto — BERTimbau contra as baselines.
    """
    nomes = list(modelos.keys())
    series = [
        (_t('Validação cruzada'), [modelos[n]['cv_f1'] for n in nomes],
         [modelos[n]['cv_f1_std'] for n in nomes], 1.00, None),
        (_t('Teste retido'), [modelos[n]['test_f1'] for n in nomes], None, 0.72, None),
        (_t('Teste, sem análogo próximo'),
         [modelos[n]['test_f1_sem_analogo'] for n in nomes], None, 0.45, '//'),
    ]

    x = np.arange(len(nomes))
    largura = 0.27

    # Achatada de propósito: os valores ficam todos acima de 0,8 e o eixo começa
    # em zero (sem truncar), então altura menor evita uma faixa vazia no PDF.
    fig, ax = plt.subplots(figsize=(8.4, 3.5))
    for k, (rotulo, valores, erros, alfa, hachura) in enumerate(series):
        cores = [COR_BERT if n == 'BERTimbau' else COR_BASE for n in nomes]
        barras = ax.bar(x + (k - 1) * largura, valores, largura,
                        yerr=erros, capsize=3, label=rotulo,
                        color=cores, alpha=alfa, edgecolor='white', hatch=hachura)
        for barra, valor in zip(barras, valores):
            ax.annotate(_virgula(valor),
                        xy=(barra.get_x() + barra.get_width() / 2, valor),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7.5)

    todos = [v for _, valores, _, _, _ in series for v in valores]
    ax.set_ylabel(_t('F1 (macro)'), fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(nomes, fontsize=9.5)
    ax.set_ylim(0, max(todos) * 1.22)
    ax.set_yticklabels([_virgula(t, 1) for t in ax.get_yticks()])
    ax.legend(fontsize=8.5, ncol=3, loc='upper center', framealpha=0.9)
    ax.grid(axis='y', alpha=0.25)
    return _finalizar(fig, ax, 'fig-tox-comparativo-f1.png')


def fig_matriz_confusao(cm):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    im = ax.imshow(cm, cmap='Blues')

    limite = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm[i, j] / cm[i].sum() * 100 if cm[i].sum() else 0
            ax.text(j, i, f'{cm[i, j]}\n{_virgula(pct, 1)}%', ha='center', va='center',
                    fontsize=10, color='white' if cm[i, j] > limite else '#2c3e50')

    ax.set_xticks(range(len(CLASSES)), _classes_exibicao())
    ax.set_yticks(range(len(CLASSES)), _classes_exibicao())
    ax.set_xlabel(_t('Classe predita'), fontsize=11)
    ax.set_ylabel(_t('Classe real'), fontsize=11)
    ax.spines[['top', 'right']].set_visible(True)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    caminho = os.path.join(OUTPUT_DIR, 'fig-tox-matriz-confusao.png')
    fig.tight_layout()
    fig.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  fig-tox-matriz-confusao.png')
    return caminho


def fig_metricas_classe(precisao, recall, f1):
    x = np.arange(len(CLASSES))
    largura = 0.26

    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    series = [(_t('Precisão'), precisao, '#3498db'),
              (_t('Recall'), recall, '#9b59b6'),
              (_t('F1'), f1, '#16a085')]
    for k, (rotulo, valores, cor) in enumerate(series):
        barras = ax.bar(x + (k - 1) * largura, valores, largura,
                        label=rotulo, color=cor, edgecolor='white')
        for barra, valor in zip(barras, valores):
            ax.annotate(_virgula(valor * 100, 1),
                        xy=(barra.get_x() + barra.get_width() / 2, valor),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(_classes_exibicao(), fontsize=10)
    ax.set_ylabel(_t('Score'), fontsize=11)
    ax.set_ylim(0, 1.13)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticklabels([_virgula(t, 1) for t in np.arange(0, 1.01, 0.2)])
    ax.legend(fontsize=9, ncol=3, loc='upper center', framealpha=0.9)
    ax.grid(axis='y', alpha=0.25)
    return _finalizar(fig, ax, 'fig-tox-metricas-classe.png')


def fig_f1_por_classe_comparativo(f1_por_modelo):
    """F1 por classe: BERTimbau contra as duas melhores baselines."""
    nomes = list(f1_por_modelo.keys())
    x = np.arange(len(CLASSES))
    largura = 0.8 / len(nomes)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for k, nome in enumerate(nomes):
        valores = f1_por_modelo[nome]
        deslocamento = (k - (len(nomes) - 1) / 2) * largura
        cor = COR_BERT if nome == 'BERTimbau' else CORES[k % len(CORES)]
        barras = ax.bar(x + deslocamento, valores, largura, label=nome,
                        color=cor, edgecolor='white',
                        alpha=1.0 if nome == 'BERTimbau' else 0.75)
        for barra, valor in zip(barras, valores):
            ax.annotate(_virgula(valor * 100, 1),
                        xy=(barra.get_x() + barra.get_width() / 2, valor),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(_classes_exibicao(), fontsize=10)
    ax.set_ylabel(_t('F1 por classe'), fontsize=11)
    ax.set_ylim(0, 1.13)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticklabels([_virgula(t, 1) for t in np.arange(0, 1.01, 0.2)])
    ax.legend(fontsize=9, ncol=len(nomes), loc='upper center', framealpha=0.9)
    ax.grid(axis='y', alpha=0.25)
    return _finalizar(fig, ax, 'fig-tox-f1-por-classe-comparativo.png')


# Marcas e canais que aparecem no título mas não identificam o vídeo
MARCAS = ('looney tunes cartoons', 'looney tunes', 'cartoon network',
          'adult swim', 'wb kids', 'em português', 'brasil')


def fig_score_por_classe(distribuicao):
    """
    Distribuição do \emph{score} de segurança por classe anotada, no teste retido.

    Substitui a demonstração com um punhado de vídeos: mostrar que as três
    classes ocupam faixas distintas do \emph{score} é o que sustenta a afirmação
    de que a reordenação separa conteúdo.
    """
    dados = [distribuicao[c]['scores'] for c in CLASSES]

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    partes = ax.violinplot(dados, positions=range(len(CLASSES)), widths=0.75,
                           showextrema=False, showmedians=False)
    for corpo, cor in zip(partes['bodies'], CORES):
        corpo.set_facecolor(cor)
        corpo.set_alpha(0.35)
        corpo.set_edgecolor('white')

    caixa = ax.boxplot(dados, positions=range(len(CLASSES)), widths=0.16,
                       patch_artist=True, showfliers=False,
                       medianprops=dict(color='#2c3e50', linewidth=1.6))
    for patch, cor in zip(caixa['boxes'], CORES):
        patch.set_facecolor(cor)
        patch.set_alpha(0.9)
        patch.set_edgecolor('white')

    for i, classe in enumerate(CLASSES):
        d = distribuicao[classe]
        ax.annotate(f"{_t('mediana')} {_virgula(d['mediana'], 2)}\n(n={d['n']})",
                    xy=(i, 1.03), ha='center', va='bottom', fontsize=8.5,
                    color='#2c3e50')

    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(_classes_exibicao(), fontsize=10)
    ax.set_xlabel(_t('Classe anotada'), fontsize=11)
    ax.set_ylabel(_t('Score de segurança'), fontsize=11)
    ax.set_ylim(-0.03, 1.16)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticklabels([_virgula(t, 1) for t in np.arange(0, 1.01, 0.2)])
    ax.grid(axis='y', alpha=0.25)
    return _finalizar(fig, ax, 'fig-tox-score-por-classe.png')


def fig_sem_finetuning(referencias):
    """Barras horizontais: o que se obtém sem ajustar o modelo neste corpus."""
    itens = sorted(referencias.items(), key=lambda kv: kv[1])
    chaves = [n for n, _ in itens]
    valores = [v for _, v in itens]

    # A chave interna e sempre em portugues; o rotulo exibido segue o idioma e
    # perde a marcacao LaTeX, que o matplotlib imprimiria literalmente.
    nomes = [_sem_latex(_t(n)) for n in chaves]
    cores = [COR_BERT if 'ajuste fino' in n else COR_BASE for n in chaves]

    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    barras = ax.barh(range(len(nomes)), valores, 0.62, color=cores, edgecolor='white')
    for barra, valor in zip(barras, valores):
        ax.annotate(_virgula(valor), xy=(valor, barra.get_y() + barra.get_height() / 2),
                    xytext=(4, 0), textcoords='offset points', va='center', fontsize=9)

    ax.set_yticks(range(len(nomes)))
    ax.set_yticklabels(nomes, fontsize=9)
    ax.set_xlabel(_t('F1 macro no teste retido'), fontsize=11)
    ax.set_xlim(0, 1.06)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xticklabels([_virgula(t, 1) for t in np.arange(0, 1.01, 0.2)])
    ax.grid(axis='x', alpha=0.25)
    return _finalizar(fig, ax, 'fig-tox-sem-finetuning.png')


# ----------------------------------------------------------------------------
# Trecho .tex
# ----------------------------------------------------------------------------

def _num(valor, casas=4):
    """Formata número para o LaTeX no separador decimal do idioma do artigo."""
    texto = f'{valor:.{casas}f}'
    return texto if IDIOMA == 'en' else texto.replace('.', '{,}')


CABECALHO_TEX = (
    '% Gerado por app/nlp/evaluation/scripts/gerar_figuras_artigo.py\n'
    '% NAO EDITAR A MAO: reexecute o script apos retreinar.\n'
)


def _salvar_tex(linhas, caminho):
    with open(caminho, 'w', encoding='utf-8') as f:
        f.write(CABECALHO_TEX + '\n'.join(linhas) + '\n')
    print(f"  {os.path.basename(caminho)}")


def escrever_tex(numeros, destino_dir):
    """
    Escreve uma tabela por arquivo, para que o main.tex as inclua com \\input
    e não contenha nenhum número digitado à mão.
    """
    stage2 = numeros['halving_top6']
    linhas = []
    add = linhas.append

    add(_abre_tabela())
    add('\\centering')
    if IDIOMA == 'en':
        add('\\caption{Stage 2 of the search: configurations promoted from screening, '
            'evaluated with 5-fold cross-validation. LLRD is the layer-wise learning '
            'rate decay factor; \\emph{Over}.\\ indicates whether the training split of '
            'the fold was balanced by oversampling. CE stands for cross-entropy.}')
    else:
        add('\\caption{Estágio 2 da busca: configurações promovidas da triagem, '
            'avaliadas com validação cruzada de 5 \\emph{folds}. LLRD é o fator de '
            'decaimento da taxa de aprendizado por camada; \\emph{Over}.\\ indica se o '
            'treino do \\emph{fold} foi balanceado por \\emph{oversampling}.}')
    add('\\label{tab:halving}')
    add('\\begin{tabular}{ccccccc}')
    add('\\hline')
    if IDIOMA == 'en':
        add('Epochs & Batch & Learn.\\ rate & LLRD & Loss & \\emph{Over}. & F1 (macro) \\\\')
    else:
        add('Épocas & Lote & Taxa Aprend. & LLRD & Perda & \\emph{Over}. & F1 (macro) \\\\')
    add('\\hline')
    for i, cfg in enumerate(stage2):
        celulas = [
            str(cfg['epochs']), str(cfg['batch_size']),
            f"{cfg['learning_rate']:g}".replace('e-0', 'e-'),
            _num(cfg['llrd'], 2),
            'Focal' if cfg['loss_type'] == 'focal' else ('CE' if IDIOMA == 'en' else 'EC'),
            ('yes' if IDIOMA == 'en' else 'sim') if cfg['oversample']
            else ('no' if IDIOMA == 'en' else 'não'),
            f"{_num(cfg['mean_f1'])} $\\pm$ {_num(cfg['std_f1'], 3)}",
        ]
        if i == 0:
            celulas = [f'\\textbf{{{c}}}' for c in celulas]
        add(' & '.join(celulas) + ' \\\\')
    add('\\hline')
    add('\\end{tabular}')
    add(_fecha_tabela())
    _salvar_tex(linhas, os.path.join(destino_dir, 'tab-halving.tex'))

    linhas = []
    add = linhas.append
    n_com = numeros['similaridade']['n_com_analogo']
    n_sem = numeros['similaridade']['n_sem_analogo']
    limiar = _num(numeros['similaridade']['limiar'], 1)
    add(_abre_tabela())
    add('\\centering')
    if IDIOMA == 'en':
        add('\\caption{Macro F1 comparison: TF-IDF baselines vs.\\ BERTimbau. The last '
            'two columns split the held-out test set into the sentences that do and do '
            'not have a near analogue in the training data (character $n$-gram cosine '
            f'above {limiar}): {n_com} and {n_sem} sentences, respectively.}}')
    else:
        add('\\caption{Comparação de F1 (macro): \\emph{baselines} TF-IDF vs.\\ BERTimbau. '
            'As duas últimas colunas separam o conjunto de teste entre as frases que têm '
            f'e as que não têm um análogo próximo no treino (cosseno de $n$-gramas de '
            f'caracteres acima de {limiar}): {n_com} e {n_sem} frases, respectivamente.}}')
    add('\\label{tab:baselines}')
    add('\\begin{tabular}{lcccc}')
    add('\\hline')
    if IDIOMA == 'en':
        add(' & & & \\multicolumn{2}{c}{Test F1, by proximity} \\\\')
        add('\\cline{4-5}')
        add('Model & F1 (cross-val.) & F1 (test) & with analogue & without analogue \\\\')
    else:
        add(' & & & \\multicolumn{2}{c}{F1 no teste, por proximidade} \\\\')
        add('\\cline{4-5}')
        add('Modelo & F1 (val.\\ cruzada) & F1 (teste) & com análogo & sem análogo \\\\')
    add('\\hline')
    for nome, dados in numeros['modelos'].items():
        celulas = [
            nome,
            f"{_num(dados['cv_f1'])} $\\pm$ {_num(dados['cv_f1_std'], 3)}",
            _num(dados['test_f1']),
            _num(dados['test_f1_com_analogo']),
            _num(dados['test_f1_sem_analogo']),
        ]
        if nome == 'BERTimbau':
            celulas = [f'\\textbf{{{c}}}' for c in celulas]
        add(' & '.join(celulas) + ' \\\\')
    add('\\hline')
    add('\\end{tabular}')
    add(_fecha_tabela())
    _salvar_tex(linhas, os.path.join(destino_dir, 'tab-baselines.tex'))

    if numeros.get('sem_finetuning'):
        linhas = []
        add = linhas.append
        add(_abre_tabela())
        add('\\centering')
        if IDIOMA == 'en':
            add('\\caption{References that do not use the labels of this corpus for '
                'training, on the held-out test set. The LLM operates few-shot, with two '
                'examples per class drawn from the training split; the zero-shot row '
                'queries BERTimbau by masked-token filling, without supervision; the '
                'off-the-shelf models are public Portuguese toxicity classifiers, '
                'composed into the three-class scheme of this work.}')
        else:
            add('\\caption{Referências que não utilizam os rótulos deste corpus para '
                'treinar, no teste retido. O LLM opera em \\emph{few-shot}, com dois '
                'exemplos por classe retirados do treino; o zero-shot consulta o '
                'BERTimbau por preenchimento de máscara, sem supervisão; os modelos '
                'prontos são classificadores públicos de toxicidade em português, '
                'compostos no esquema de três classes deste trabalho.}')
        add('\\label{tab:sem-finetuning}')
        add('\\begin{tabular}{lc}')
        add('\\hline')
        add(('Reference & F1 (macro) \\\\' if IDIOMA == 'en'
             else 'Referência & F1 (macro) \\\\'))
        add('\\hline')
        for nome, valor in numeros['sem_finetuning']:
            celulas = [_t(nome), _num(valor)]
            if 'ajuste fino' in nome:
                celulas = [f'\\textbf{{{c}}}' for c in celulas]
            add(' & '.join(celulas) + ' \\\\')
        for nome, valor in numeros.get('sem_finetuning_binarias', {}).items():
            add(f'\\hline')
            break
        for nome, valor in numeros.get('sem_finetuning_binarias', {}).items():
            add(f'\\emph{{{_t(nome)}}} & {_num(valor)} \\\\')
        add('\\hline')
        add('\\end{tabular}')
        add(_fecha_tabela())
        _salvar_tex(linhas, os.path.join(destino_dir, 'tab-sem-finetuning.tex'))

    linhas = []
    add = linhas.append
    add(_abre_tabela())
    add('\\centering')
    add('\\caption{%s}' % ('Per-class metrics on the held-out test set.' if IDIOMA == 'en'
                           else 'Métricas por classe de toxicidade (conjunto de teste retido).'))
    add('\\label{tab:classes}')
    add('\\begin{tabular}{lrrrr}')
    add('\\hline')
    add(('Class & Precision & Recall & F1 & Support \\\\' if IDIOMA == 'en'
         else 'Classe & Precisão & \\emph{Recall} & F1 & Suporte \\\\'))
    add('\\hline')
    for i, classe in enumerate(CLASSES):
        pct = lambda chave: _num(numeros['bert_teste'][chave][i] * 100, 1) + '\\%'
        add(f"{_t(classe)} & {pct('precision_per_class')} & {pct('recall_per_class')} "
            f"& {pct('f1_per_class')} & {numeros['bert_teste']['support_per_class'][i]} \\\\")
    add('\\hline')
    add('\\end{tabular}')
    add(_fecha_tabela())
    _salvar_tex(linhas, os.path.join(destino_dir, 'tab-classes.tex'))


# ----------------------------------------------------------------------------

def main():
    global IDIOMA, DUAS_COLUNAS

    parser = argparse.ArgumentParser(description='Gera figuras e tabelas do artigo')
    parser.add_argument('--copiar-para', default=None,
                        help='Diretório de figuras do LaTeX para receber os PNGs')
    parser.add_argument('--idioma', choices=['pt', 'en'], default='pt',
                        help='Idioma dos rótulos, legendas e tabelas (padrão: pt)')
    parser.add_argument('--duas-colunas', action='store_true',
                        help='Emite table* (largura da página) em vez de table, '
                             'para templates de duas colunas como o do JIS')
    args = parser.parse_args()

    IDIOMA = args.idioma
    DUAS_COLUNAS = args.duas_colunas
    print(f"Idioma de saída: {IDIOMA} | duas colunas: {DUAS_COLUNAS}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    corpus, arq_corpus = _carregar(os.path.join(RESULTS_DIR, 'corpus', 'corpus_stats.json'))
    grid, arq_grid = _carregar(
        os.path.join(RESULTS_DIR, 'toxicity_gridsearch', 'gridsearch_optimized_*.json'))
    avaliacao, arq_aval = _carregar(
        os.path.join(RESULTS_DIR, 'toxicity_evaluation', 'evaluation_metrics_TOX_*.json'))
    baselines, arq_base = _carregar(
        os.path.join(RESULTS_DIR, 'baselines_toxicity_*.json'))
    scores_exemplo, _ = _carregar(
        os.path.join(OUTPUT_DIR, 'scores_exemplo.json'), obrigatorio=False)
    sem_ft, arq_sem_ft = _carregar(
        os.path.join(RESULTS_DIR, 'sem_finetuning_*.json'), obrigatorio=False)
    pareada, arq_pareada = _carregar(
        os.path.join(RESULTS_DIR, 'comparacao_pareada_*.json'), obrigatorio=False)
    llm, arq_llm = _carregar(
        os.path.join(RESULTS_DIR, 'llm_baselines_toxicity_*.json'), obrigatorio=False)

    print(f"Fontes:\n  corpus:    {arq_corpus}\n  gridsearch:{arq_grid}\n"
          f"  avaliação: {arq_aval}\n  baselines: {arq_base}\n")

    # --- números do BERTimbau ---
    melhor = grid['best_config']
    stage2 = sorted(grid['stage2']['results'], key=lambda r: r['mean_f1'], reverse=True)
    halving_top6 = [{
        'epochs': r['config']['epochs'],
        'batch_size': r['config']['batch_size'],
        'learning_rate': float(r['config']['learning_rate']),
        'llrd': float(r['config'].get('llrd', 1.0)),
        'loss_type': r['config'].get('loss_type', 'focal'),
        'oversample': bool(r['config'].get('oversample', True)),
        'mean_f1': r['mean_f1'],
        'std_f1': r['std_f1'],
        'mean_accuracy': r.get('mean_accuracy'),
    } for r in stage2]

    # No modo 'search' a vencedora sai do estágio de estabilidade, que agrega
    # sementes e não guarda acurácia; recupera-se a do estágio 2 quando existir.
    if melhor is not None and 'mean_accuracy' not in melhor:
        igual = next((r for r in stage2 if r['config'] == melhor['config']), None)
        melhor = dict(melhor, mean_accuracy=(igual or {}).get('mean_accuracy'))

    bert_teste = {
        'accuracy': avaliacao['accuracy'],
        'f1_macro': avaliacao['f1_macro'],
        'f1_weighted': avaliacao['f1_weighted'],
        'precision_macro': avaliacao['precision_macro'],
        'recall_macro': avaliacao['recall_macro'],
        'precision_per_class': avaliacao['precision_per_class'],
        'recall_per_class': avaliacao['recall_per_class'],
        'f1_per_class': avaliacao['f1_per_class'],
        'support_per_class': avaliacao['support_per_class'],
        'confusion_matrix': avaliacao['confusion_matrix'],
        'por_sentenca_distinta': avaliacao['por_sentenca_distinta'],
        'por_similaridade': avaliacao['por_similaridade'],
    }

    sim_bert = avaliacao['por_similaridade']

    # --- tabela unificada de modelos, BERTimbau primeiro ---
    modelos = {'BERTimbau': {
        'cv_f1': melhor['mean_f1'],
        'cv_f1_std': melhor['std_f1'],
        'test_f1': bert_teste['f1_macro'],
        'test_f1_com_analogo': sim_bert['com_analogo']['f1_macro'],
        'test_f1_sem_analogo': sim_bert['sem_analogo']['f1_macro'],
        'f1_per_class': bert_teste['f1_per_class'],
    }}
    ordenadas = sorted(baselines['cross_validation_results'].items(),
                       key=lambda kv: kv[1]['f1_mean'], reverse=True)
    for nome, cv in ordenadas:
        sim = baselines['test_results'][nome]['por_similaridade']
        modelos[nome] = {
            'cv_f1': cv['f1_mean'],
            'cv_f1_std': cv['f1_std'],
            'test_f1': baselines['test_results'][nome]['f1'],
            'test_f1_com_analogo': sim['com_analogo']['f1'],
            'test_f1_sem_analogo': sim['sem_analogo']['f1'],
            'f1_per_class': baselines['test_results'][nome]['f1_per_class'],
        }

    # Referências que não usam os rótulos deste corpus, para a figura e a tabela
    # próprias. A sonda linear fica de fora: ela é treinada nos nossos rótulos e
    # portanto não pertence a este bloco.
    referencias_sem_ft = {'BERTimbau (ajuste fino)': bert_teste['f1_macro']}
    if sem_ft:
        for r in sem_ft['resultados']:
            nome = r['modelo']
            if nome.startswith('Sonda linear'):
                continue
            if nome.startswith('Zero-shot máscara'):
                nome = 'Zero-shot por máscara (melhor molde)'
                referencias_sem_ft[nome] = max(referencias_sem_ft.get(nome, 0), r['f1_macro'])
            elif 'isolado' in nome:
                # Tarefas binárias: reportadas à parte, na sua própria escala
                referencias_sem_ft.setdefault('_binarias', {})[nome] = r['f1_macro']
            else:
                referencias_sem_ft[nome] = r['f1_macro']

    binarias = referencias_sem_ft.pop('_binarias', {})

    # LLMs generativos em few-shot: também não usam os rótulos deste corpus
    # para treinar, então entram no mesmo bloco de referências.
    llm_sem_analogo = {}
    if llm:
        for nome, r in llm['results'].items():
            referencias_sem_ft[f'{nome} (\\emph{{few-shot}})'] = r['f1']
            sim = r.get('por_similaridade', {}).get('sem_analogo')
            if sim:
                llm_sem_analogo[nome] = sim['f1']

    melhor_baseline = ordenadas[0][0]
    ganho_abs = bert_teste['f1_macro'] - modelos[melhor_baseline]['test_f1']
    ganho_sem_analogo = (modelos['BERTimbau']['test_f1_sem_analogo']
                         - modelos[melhor_baseline]['test_f1_sem_analogo'])

    cm = np.array(bert_teste['confusion_matrix'])
    toxicos_como_seguro = int(cm[1, 0] + cm[2, 0])
    seguro_como_toxico = int(cm[0, 1] + cm[0, 2])
    confusao_leve_severa = int(cm[1, 2] + cm[2, 1])

    numeros = {
        'fontes': {'corpus': arq_corpus, 'gridsearch': arq_grid,
                   'avaliacao': arq_aval, 'baselines': arq_base},
        'corpus': corpus,
        'split': {
            'treino_cv': baselines['methodology']['train_size'],
            'teste': baselines['methodology']['test_size'],
            'total': baselines['methodology']['corpus_total'],
        },
        'melhor_config': melhor['config'],
        'bert_cv': {'f1_macro': melhor['mean_f1'], 'std': melhor['std_f1'],
                    'accuracy': melhor['mean_accuracy']},
        'bert_teste': bert_teste,
        'halving_top6': halving_top6,
        'modelos': modelos,
        'melhor_baseline': melhor_baseline,
        'ganho_absoluto_vs_melhor_baseline': ganho_abs,
        'ganho_relativo_vs_melhor_baseline': ganho_abs / modelos[melhor_baseline]['test_f1'],
        'ganho_sem_analogo_vs_melhor_baseline': ganho_sem_analogo,
        'sem_finetuning': sorted(referencias_sem_ft.items(), key=lambda kv: -kv[1]),
        'sem_finetuning_binarias': binarias,
        'llm_sem_analogo': llm_sem_analogo,
        'llm_bruto': llm,
        'comparacao_pareada': pareada,
        'similaridade': {
            'limiar': sim_bert['limiar'],
            'n_com_analogo': sim_bert['com_analogo']['n'],
            'n_sem_analogo': sim_bert['sem_analogo']['n'],
            'pct_sem_analogo': sim_bert['sem_analogo']['n'] / (
                sim_bert['com_analogo']['n'] + sim_bert['sem_analogo']['n']),
        },
        'erros': {
            'toxico_classificado_como_seguro': toxicos_como_seguro,
            'total_toxicos_no_teste': int(cm[1].sum() + cm[2].sum()),
            'seguro_classificado_como_toxico': seguro_como_toxico,
            'total_seguros_no_teste': int(cm[0].sum()),
            'confusao_leve_severa': confusao_leve_severa,
            'leve_predita_severa': int(cm[1, 2]),
            'severa_predita_leve': int(cm[2, 1]),
        },
    }

    print("Figuras:")
    fig_distribuicao_corpus(corpus)
    fig_comparativo_f1(modelos)
    fig_matriz_confusao(bert_teste['confusion_matrix'])
    fig_metricas_classe(bert_teste['precision_per_class'],
                        bert_teste['recall_per_class'],
                        bert_teste['f1_per_class'])
    top3 = ['BERTimbau'] + [n for n, _ in ordenadas[:2]]
    fig_f1_por_classe_comparativo({n: modelos[n]['f1_per_class'] for n in top3})
    if scores_exemplo:
        fig_score_por_classe(scores_exemplo['distribuicao_por_classe'])
        numeros['scores_exemplo'] = scores_exemplo
    if sem_ft:
        fig_sem_finetuning(referencias_sem_ft)

    print("Tabelas:")
    escrever_tex(numeros, OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, 'resumo_numeros.json'), 'w', encoding='utf-8') as f:
        json.dump(numeros, f, indent=2, ensure_ascii=False)
    print('  resumo_numeros.json')

    if args.copiar_para:
        destino = os.path.abspath(args.copiar_para)
        os.makedirs(destino, exist_ok=True)
        for png in glob.glob(os.path.join(OUTPUT_DIR, '*.png')):
            shutil.copy2(png, destino)
        print(f"\nFiguras copiadas para: {destino}")

        # As tabelas ficam ao lado de figuras/, em tabelas/, para o \input
        destino_tab = os.path.join(os.path.dirname(destino), 'tabelas')
        os.makedirs(destino_tab, exist_ok=True)
        for tex in glob.glob(os.path.join(OUTPUT_DIR, 'tab-*.tex')):
            shutil.copy2(tex, destino_tab)
        print(f"Tabelas copiadas para: {destino_tab}")

    print(f"\nSaída em: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
