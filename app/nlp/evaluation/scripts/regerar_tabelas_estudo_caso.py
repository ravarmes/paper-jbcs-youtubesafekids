"""
Regera as saidas LaTeX do estudo de caso e da simulacao a partir dos JSON de
resultado ja salvos, sem precisar do modelo treinado.

Tambem e o modulo UNICO das funcoes que montam essas tabelas e figuras:
`analisar_estratos.py` e `simular_amostragem.py` importam daqui, de modo que o
texto de cada tabela e a semantica de cada figura tenham uma so fonte. Nenhum
valor e digitado a mao: tudo sai dos JSON.

Entradas (as mais recentes da pasta de resultados da rodada, ou de --resultados):
  estratos_*.json
  simulacao_amostragem_*.json

Saidas (em experimento.dir_latex(...), ou em <--saida>/latex/... quando dado):
  tabelas/tab-estratos.tex
  tabelas/tab-limiares-estudo-caso.tex   desfecho por limiar, regra explicita (M4b, M4c)
  tabelas/tab-simulacao.tex              deteccao simulada com IC 95% (M5b), OOF e teste
  tabelas/tab-simulacao-variantes.tex    variantes da simulacao (M5a, M5c)
  figuras/fig11-estratos.png             score so de toxicidade, linha no limiar a priori
  figuras/fig12-ordem.png                duas cores no limiar designado (m8)
  figuras/fig12-ordem-banda.png          a mesma figura com a banda de tolerancia (M7c)
  (com --idioma pt as figuras levam o sufixo -pt)

Uso:
    python -m app.nlp.evaluation.scripts.regerar_tabelas_estudo_caso --idioma en
    python -m app.nlp.evaluation.scripts.regerar_tabelas_estudo_caso --idioma en \
        --resultados <pasta com os JSON> --saida <pasta de saida>
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from app.nlp import experimento  # noqa: E402

# Rotulo de cada regra de agregacao, na ordem em que aparece na tabela.
AGREGACOES = [
    ('media', 'Média das $k$ (Eq.\\ atual)', 'Mean of the $k$ (current Eq.)'),
    ('minimo', 'Mínimo das $k$', 'Minimum of the $k$'),
    ('min_ponderado', '$0{,}7\\cdot$mín $+\\ 0{,}3\\cdot$média',
     '$0.7\\cdot$min $+\\ 0.3\\cdot$mean'),
    ('duas_piores', 'Média das 2 piores', 'Mean of the 2 lowest'),
]

KS = ['3', '5', '8', '12']

# Ordem e rotulo das populacoes de amostragem da simulacao.
POPULACOES = [
    ('oof', 'predições \\emph{out-of-fold} sobre a partição de treino',
     'out-of-fold predictions on the training split'),
    ('teste', 'teste retido', 'held-out test set'),
]

# Ordem e rotulo das variantes da simulacao (simular_amostragem.VARIANTES).
VARIANTES = [
    ('iid_nenhuma', 'i.i.d.; parte apropriada: Nenhuma',
     'i.i.d.; appropriate part: None'),
    ('iid_nenhuma_leve', 'i.i.d.; parte apropriada: Nenhuma e Leve',
     'i.i.d.; appropriate part: None and Mild'),
    ('bloco_fixo', 'bloco contíguo; posições fixas',
     'contiguous block; fixed positions'),
    ('bloco_estratificado', 'bloco contíguo; uma posição sorteada por estrato',
     'contiguous block; one random position per stratum'),
]

# O template do JIS/JBCS e de duas colunas: as tabelas saem como table*,
# que ocupa a largura total da pagina.
DUAS_COLUNAS = True

# Semaforo do artigo (ver gerar_figuras_artigo.py): verde = nao rebaixado,
# vermelho escuro = rebaixado. A luminosidade tambem separa as duas classes.
COR_A, COR_B = '#2ecc71', '#e74c3c'
VERDE, VERMELHO = '#6fc48f', '#7a1616'
CINZA_LIMIAR = '#7f8c8d'


def _abre_tabela():
    return '\\begin{table*}[!t]' if DUAS_COLUNAS else '\\begin{table}[!t]'


def _fecha_tabela():
    return '\\end{table*}' if DUAS_COLUNAS else '\\end{table}'


def _mais_recente(pasta, padrao):
    arquivos = glob.glob(os.path.join(str(pasta), padrao))
    if not arquivos:
        raise FileNotFoundError(f'Nenhum arquivo casa com {padrao} em {pasta}')
    return max(arquivos, key=os.path.getmtime)


def _carregar(pasta, padrao):
    caminho = _mais_recente(pasta, padrao)
    with open(caminho, 'r', encoding='utf-8') as f:
        return json.load(f), os.path.basename(caminho)


def destinos(saida=None):
    """Pastas de tabelas e figuras: as da rodada, ou <saida>/latex/... (validação)."""
    if saida:
        tab = Path(saida) / 'latex' / 'tabelas'
        fig = Path(saida) / 'latex' / 'figuras'
    else:
        tab = experimento.dir_latex('tabelas')
        fig = experimento.dir_latex('figuras')
    tab.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    return tab, fig


def _dec(texto, idioma):
    return texto if idioma == 'en' else texto.replace('.', '{,}')


def _dec_txt(texto, idioma):
    """Separador decimal para rotulos de figura (texto puro, sem LaTeX)."""
    return texto if idioma == 'en' else texto.replace('.', ',')


def limiar_designado(dados):
    """
    Limiar usado nas figuras e na tabela de limiares: o fixado a priori pela
    simulação (bloco `limiar_designado` do JSON de estratos). JSONs antigos não o
    têm; neles vale o limiar da análise de falhas (0,85 na versão submetida).
    Devolve (valor, origem, populacao).
    """
    ld = dados.get('limiar_designado') or {}
    if ld.get('valor') is not None:
        return float(ld['valor']), ld.get('origem', 'a_priori'), ld.get('populacao')
    fb = dados.get('falhas_estrato_b') or {}
    return float(fb.get('limiar', 0.85)), 'legado', None


# =============================================================================
# Tabelas do estudo de caso
# =============================================================================

def tabela_estratos(dados, idioma):
    if idioma == 'en':
        legenda = (
            '\\caption{Distribution of the video verbal-toxicity score (toxicity filter only: '
            'mean of the scores of the three sampled sentences) across the four queries '
            'of the case study, ten videos each. In stratum~A the desirable outcome is '
            'that scores be high and close to one another; in stratum~B, that they be '
            'low. Queries are reproduced verbatim in Brazilian Portuguese because they '
            'are inputs to the system; English translations are given in the text.}')
        cabecalho = 'Stratum & Query & $n$ & Mean & Minimum & Range \\\\'
        agregado = 'aggregate'
    else:
        legenda = (
            '\\caption{Distribuição do \\emph{score} de segurança do vídeo (só o filtro '
            'de toxicidade: média dos \\emph{scores} das três frases amostradas) nas '
            'quatro buscas do estudo de caso, com dez vídeos cada. No estrato A o '
            'desejável é que os \\emph{scores} sejam altos e próximos entre si; no B, '
            'que sejam baixos.}')
        cabecalho = 'Estrato & Termo de busca & $n$ & Média & Mínimo & Amplitude \\\\'
        agregado = 'agregado'

    nomes = {'A_responsavel': 'A', 'B_apelo_infantil': 'B'}
    linhas = [_abre_tabela(), '\\centering', legenda,
              '\\label{tab:estratos}',
              '\\begin{tabular}{llcccc}', '\\toprule', cabecalho, '\\midrule']

    for termo, d in dados['por_termo'].items():
        seguro = termo.replace('&', '\\&')
        linhas.append(_dec(
            f"{nomes[d['estrato']]} & \\emph{{{seguro}}} & {d['n']} & "
            f"{d['media']:.3f} & {d['min']:.3f} & {d['amplitude']:.3f} \\\\", idioma))

    linhas.append('\\midrule')
    for chave, rot in (('estrato_A_responsavel', 'A'), ('estrato_B_apelo_infantil', 'B')):
        d = dados[chave]
        linhas.append(_dec(
            f"\\textbf{{{rot}}} & \\emph{{{agregado}}} & {d['n']} & "
            f"\\textbf{{{d['media']:.3f}}} & {d['min']:.3f} & {d['amplitude']:.3f} \\\\",
            idioma))

    linhas += ['\\bottomrule', '\\end{tabular}', _fecha_tabela(), '']
    return '\n'.join(linhas)


def tabela_estrato_b(dados, idioma):
    """
    Os vídeos do estrato B, um por linha (M4d: todas as perdas nomeadas, no limiar
    designado). Para cada vídeo: busca, posição na plataforma, título, a classe
    predita de cada frase amostrada (início, meio, fim), o score e o desfecho no
    limiar designado. A categoria da perda é MECÂNICA, a mesma de
    analisar_estratos (por_categoria_provisoria): "diluição" quando alguma frase
    foi predita Leve ou Severa mas a média ficou acima do limiar; "nenhuma frase
    sinalizada" quando as três foram preditas Nenhuma. Não é juízo humano sobre o
    conteúdo das frases.

    Com os rótulos da anotação cega das frases (analisar_estratos --rotulos-humanos),
    a tabela ganha a coluna dos anotadores e a categoria passa a ser a FINAL de
    analisar_estratos.classificar_perda: diluição, erro do classificador ou nenhuma
    frase tóxica (para os anotadores).
    """
    tau, _, _ = limiar_designado(dados)
    tau_txt = _dec(f'{tau:.3f}', idioma)
    en = idioma == 'en'
    letra = ({'Nenhuma': 'N', 'Leve': 'M', 'Severa': 'S'} if en
             else {'Nenhuma': 'N', 'Leve': 'L', 'Severa': 'S'})
    videos = [v for v in dados['videos'] if v['estrato'].startswith('B')]
    videos.sort(key=lambda v: v['score_media'])
    humano = all('humano' in f for v in videos for f in v['frases_pontuadas'])
    final = {d['id']: d.get('categoria')
             for d in (dados.get('falhas_estrato_b') or {}).get('detalhe', [])}
    humano = humano and all(final.get(v['id']) for v in videos if v['score_media'] >= tau)
    nomes_final = ({'diluicao_na_agregacao': 'missed: dilution',
                    'erro_do_classificador': 'missed: classifier error',
                    'sem_linguagem_toxica_nas_frases_amostradas': 'missed: no toxic sentence',
                    'fora_do_escopo': 'missed: out of scope'} if en else
                   {'diluicao_na_agregacao': 'perdido: diluição',
                    'erro_do_classificador': 'perdido: erro do classificador',
                    'sem_linguagem_toxica_nas_frases_amostradas': 'perdido: nenhuma frase tóxica',
                    'fora_do_escopo': 'perdido: fora do escopo'})

    def desfecho(v):
        if v['score_media'] < tau:
            return 'demoted' if en else 'rebaixado'
        if humano:
            return nomes_final[final[v['id']]]
        sinal = any(f['classe'] != 'Nenhuma' for f in v['frases_pontuadas'])
        if en:
            return 'missed: dilution' if sinal else 'missed: no sentence flagged'
        return 'perdido: diluição' if sinal else 'perdido: nenhuma frase sinalizada'

    def titulo(t, n=58):
        # Sem emojis e símbolos gráficos (a fonte do template não os tem).
        t = ''.join(c for c in t if ord(c) < 0x2000)
        t = ' '.join(t.split())
        t = t if len(t) <= n else t[:n - 1].rstrip() + '…'
        trocas = {'&': '\\&', '%': '\\%', '$': '\\$', '#': '\\#', '_': '\\_'}
        return ''.join(trocas.get(c, c) for c in t)

    if en and humano:
        legenda = (f'\\caption{{The {len(videos)} stratum-B videos of the case study, ordered by '
                   f'verbal-toxicity score. Model: the class predicted for the beginning, middle '
                   f'and end sentence; Annot.: the label the three annotators gave the same '
                   f'sentence, blind to the video and the prediction (majority; N None, M Mild, '
                   f'S Severe). Outcome at the a priori threshold $\\tau^{{*}} = {tau_txt}$ (rule: '
                   f'score $< \\tau^{{*}}$ $\\Rightarrow$ demoted). Missed videos: '
                   f'\\emph{{dilution}} when the model flagged a sentence the annotators labelled '
                   f'toxic but the mean stayed at or above $\\tau^{{*}}$; \\emph{{classifier '
                   f'error}} when the annotators labelled a sentence toxic and the model flagged '
                   f'none of them; \\emph{{no toxic sentence}} when the annotators labelled all '
                   f'three None. The labels concern the sampled sentences, not the rest of the '
                   f'video. Titles as published, truncated.}}')
        cab = 'Query & Pos. & Title & Model & Annot. & Score & Outcome \\\\'
    elif en:
        legenda = (f'\\caption{{The {len(videos)} stratum-B videos of the case study, ordered by '
                   f'verbal-toxicity score. Classes: the class predicted for the beginning, '
                   f'middle and end sentence (N None, M Mild, S Severe). Outcome at the a priori '
                   f'threshold $\\tau^{{*}} = {tau_txt}$ (rule: score $< \\tau^{{*}}$ '
                   f'$\\Rightarrow$ demoted). The categories of the missed videos are mechanical: '
                   f'\\emph{{dilution}} when some sampled sentence was predicted Mild or Severe '
                   f'but the mean stayed at or above $\\tau^{{*}}$, \\emph{{no sentence flagged}} '
                   f'when all three were predicted None. Titles as published, truncated.}}')
        cab = 'Query & Pos. & Title & Classes & Score & Outcome \\\\'
    elif humano:
        legenda = (f'\\caption{{Os {len(videos)} vídeos do estrato B do estudo de caso, em ordem '
                   f'de \\emph{{score}}. Modelo: a classe predita para a frase do início, do meio '
                   f'e do fim; Anot.: o rótulo dos três anotadores para a mesma frase, às cegas '
                   f'(maioria; N Nenhuma, L Leve, S Severa). Desfecho no limiar \\emph{{a '
                   f'priori}} $\\tau^{{*}} = {tau_txt}$ (regra: \\emph{{score}} $< \\tau^{{*}}$ '
                   f'$\\Rightarrow$ rebaixado).}}')
        cab = 'Busca & Pos. & Título & Modelo & Anot. & \\emph{Score} & Desfecho \\\\'
    else:
        legenda = (f'\\caption{{Os {len(videos)} vídeos do estrato B do estudo de caso, em ordem '
                   f'de \\emph{{score}}. Classes: a classe predita para a frase do início, do '
                   f'meio e do fim (N Nenhuma, L Leve, S Severa). Desfecho no limiar \\emph{{a '
                   f'priori}} $\\tau^{{*}} = {tau_txt}$ (regra: \\emph{{score}} $< \\tau^{{*}}$ '
                   f'$\\Rightarrow$ rebaixado). As categorias das perdas são mecânicas.}}')
        cab = 'Busca & Pos. & Título & Classes & \\emph{Score} & Desfecho \\\\'
    colunas = ('@{}llp{0.36\\textwidth}cccl@{}' if humano else '@{}llp{0.43\\textwidth}ccl@{}')
    linhas = ['% NAO EDITAR A MAO: gerado por regerar_tabelas_estudo_caso.py (tabela_estrato_b).',
              _abre_tabela(), '\\centering', '\\small', legenda, '\\label{tab:estrato-b}',
              f'\\begin{{tabular}}{{{colunas}}}', '\\toprule', cab, '\\midrule']
    for v in videos:
        classes = '/'.join(letra.get(f['classe'], '?') for f in v['frases_pontuadas'])
        if humano:
            classes += ' & ' + '/'.join(letra.get(f['humano'], '?') for f in v['frases_pontuadas'])
        termo = 'South Park' if v['termo'].startswith('south') else 'Rick and Morty'
        score = _dec('%.3f' % v['score_media'], idioma)
        linhas.append(f"{termo} & {v['posicao_youtube']} & {titulo(v['titulo'], 50 if humano else 58)}"
                      f" & {classes} & {score} & {desfecho(v)} \\\\")
    linhas += ['\\bottomrule', '\\end{tabular}', _fecha_tabela(), '']
    return '\n'.join(linhas)


def tabela_limiares_estudo_caso(dados, idioma):
    """
    Desfecho do estudo de caso em cada limiar da grade, com a regra explícita
    (score < limiar => rebaixado). O limiar fixado a priori pela simulação vem
    marcado; os demais foram avaliados nos mesmos 40 vídeos e são exploratórios.
    """
    entradas = dados['deteccao_por_limiar']
    tau, origem, populacao = limiar_designado(dados)
    ld = dados.get('limiar_designado') or {}
    n_b = entradas[0]['total_b'] if entradas else 20
    n_a = entradas[0]['total_a'] if entradas else 20
    k = ld.get('k', 3)
    fa = ld.get('falso_alarme_alvo_pct', 5)
    tau_txt = _dec(f'{tau:.3f}', idioma)

    if idioma == 'en':
        if origem == 'legado':
            fonte = ('No threshold was fixed in advance for this run; every row is '
                     'exploratory.')
        else:
            pop = ('out-of-fold predictions on the training split' if populacao == 'oof'
                   else 'the held-out test predictions (out-of-fold predictions were '
                        'not available)')
            fonte = (f'The a priori threshold $\\tau^{{*}} = {tau_txt}$ was fixed without '
                     f'using the case-study videos: it is the {fa}\\% false-alarm point of '
                     f'the mean of $k = {k}$ sentence scores over synthetic, entirely '
                     f'appropriate videos built from {pop}. Every other threshold was '
                     f'evaluated on these same {n_a + n_b} videos and is exploratory.')
        legenda = (f'\\caption{{Case-study outcome by threshold $\\tau$, under the rule '
                   f'``video score $< \\tau$ $\\Rightarrow$ demoted\'\'. Demoted in B: '
                   f'stratum-B videos below $\\tau$; false alarms in A: stratum-A videos '
                   f'below $\\tau$. {fonte}}}')
        cabecalho = (f'$\\tau$ & Status & Demoted in B (of {n_b}) & '
                     f'False alarms in A (of {n_a}) \\\\')
        st_ap, st_ex = 'a priori', 'exploratory'
    else:
        if origem == 'legado':
            fonte = 'Nenhum limiar foi fixado antes nesta rodada; todas as linhas são exploratórias.'
        else:
            pop = ('predições \\emph{out-of-fold} sobre a partição de treino'
                   if populacao == 'oof' else
                   'predições do teste retido (as \\emph{out-of-fold} não estavam disponíveis)')
            fonte = (f'O limiar \\emph{{a priori}} $\\tau^{{*}} = {tau_txt}$ foi fixado sem '
                     f'usar os vídeos do estudo de caso: é o ponto de {fa}\\% de falso alarme da '
                     f'média dos \\emph{{scores}} de $k = {k}$ frases em vídeos sintéticos '
                     f'inteiramente apropriados, montados a partir de {pop}. Os demais '
                     f'limiares foram avaliados nestes mesmos {n_a + n_b} vídeos e são exploratórios.')
        legenda = (f'\\caption{{Desfecho do estudo de caso por limiar $\\tau$, com a regra '
                   f'``\\emph{{score}} do vídeo $< \\tau$ $\\Rightarrow$ rebaixado\'\'. '
                   f'Rebaixados em B: vídeos do estrato B abaixo de $\\tau$; falsos alarmes '
                   f'em A: vídeos do estrato A abaixo de $\\tau$. {fonte}}}')
        cabecalho = (f'$\\tau$ & Situação & Rebaixados em B (de {n_b}) & '
                     f'Falsos alarmes em A (de {n_a}) \\\\')
        st_ap, st_ex = '\\emph{a priori}', 'exploratório'

    linhas = [_abre_tabela(), '\\centering', legenda, '\\label{tab:limiares-estudo-caso}',
              '\\begin{tabular}{lccc}', '\\toprule', cabecalho, '\\midrule']
    for e in entradas:
        a_priori = bool(e.get('a_priori'))
        valor = _dec(f"{e['limiar']:.3f}" if a_priori else f"{e['limiar']:.2f}", idioma)
        celulas = [f'${valor}$', st_ap if a_priori else st_ex,
                   str(e['detectados_b']), str(e['falsos_a'])]
        if a_priori:
            celulas = [f'\\textbf{{{c}}}' for c in celulas]
        linhas.append(' & '.join(celulas) + ' \\\\')
    linhas += ['\\bottomrule', '\\end{tabular}', _fecha_tabela(), '']
    return '\n'.join(linhas)


# =============================================================================
# Tabelas da simulação
# =============================================================================

def _tabela_simulacao_legado(dados, idioma, f):
    """Formato dos JSON anteriores à revisão (só 'deteccao', sem IC)."""
    n_sim = dados['n_simulacoes']
    alarme = dados['falso_alarme_alvo_pct']
    pct_toxico = int(round(float(f) * 100))
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
    linhas = [_abre_tabela(), '\\centering', legenda, '\\label{tab:simulacao}',
              '\\begin{tabular}{lcccc}', '\\toprule', cabecalho, '\\midrule']
    bloco = dados['deteccao'][f]
    for chave, rot_pt, rot_en in AGREGACOES:
        rotulo = rot_en if idioma == 'en' else rot_pt
        celulas = [f"{round(bloco[chave][k]['deteccao'] * 100)}\\%" for k in KS]
        linhas.append(' & '.join([rotulo] + celulas) + ' \\\\')
    linhas += ['\\bottomrule', '\\end{tabular}', _fecha_tabela(), '']
    return '\n'.join(linhas)


def _celula_pct(det, idioma, com_ic=True):
    texto = f"{det['deteccao'] * 100:.1f}"
    ic = det.get('ic95_bootstrap_populacao')
    if com_ic and ic and ic[0] is not None:
        sep = ', ' if idioma == 'en' else '; '
        texto += f' [{ic[0] * 100:.1f}{sep}{ic[1] * 100:.1f}]'
    return _dec(texto, idioma)


def tabela_simulacao(dados, idioma, f='0.20'):
    """
    Detecção simulada (%, uma casa) com IC 95% por bootstrap da população de
    frases, na variante de referência (frases i.i.d., parte apropriada só de
    Nenhuma), uma faixa por população de amostragem (OOF e teste).
    """
    if 'resultados' not in dados:
        return _tabela_simulacao_legado(dados, idioma, f)

    variante = 'iid_nenhuma'
    n_sim = dados['n_simulacoes']
    alarme = dados['falso_alarme_alvo_pct']
    boot = dados.get('bootstrap_populacao', {})
    n_boot = boot.get('n_reamostragens')
    pops = [p for p in POPULACOES if p[0] in dados['resultados']]

    # Meia-largura máxima do IC de Wilson (erro de Monte Carlo) entre as células
    meia = 0.0
    for chave, _, _ in pops:
        bloco = dados['resultados'][chave]['deteccao'][variante][f]
        for regra, _, _ in AGREGACOES:
            for k in KS:
                lo, hi = bloco[regra][k]['ic95_wilson_mc']
                meia = max(meia, (hi - lo) / 2 * 100)
    meia_txt = _dec(f'{meia:.1f}', idioma)
    f_txt = _dec(f'{float(f):.2f}', idioma)

    if idioma == 'en':
        ic_txt = (f'Brackets: 95\\% interval over {n_boot} bootstrap resamples of the '
                  f'sentence population (stratified by class), which carries the '
                  f'uncertainty of having a finite set of sentences'
                  if n_boot else 'No bootstrap interval was computed')
        legenda = (
            f'\\caption{{Simulated detection (\\%) of a video in which each sampled sentence '
            f'is severely toxic with probability $f = {f_txt}$, by number $k$ of sampled '
            f'sentences and aggregation rule. Sentences are drawn independently; the '
            f'appropriate part of the video is drawn from None sentences only, and Mild '
            f'sentences are not used (Table~S3 of the supplementary file relaxes both '
            f'assumptions). Each rule and each $k$ is calibrated to a {alarme}\\% false-alarm '
            f'rate over entirely appropriate synthetic videos of the same population, so '
            f'that cells are comparable. {ic_txt}; the Monte Carlo error of each point '
            f'({n_sim:,} simulated videos) is at most $\\pm${meia_txt} points.}}')
        cabecalho = 'Aggregation & $k=3$ & $k=5$ & $k=8$ & $k=12$ \\\\'
    else:
        ic_txt = (f'Entre colchetes: intervalo de 95\\% sobre {n_boot} reamostragens '
                  f'\\emph{{bootstrap}} da população de frases (estratificadas por classe), '
                  f'que carrega a incerteza de haver um número finito de frases'
                  if n_boot else 'Sem intervalo \\emph{bootstrap}')
        legenda = (
            f'\\caption{{Detecção simulada (\\%) de um vídeo em que cada frase amostrada é '
            f'severamente tóxica com probabilidade $f = {f_txt}$, por número $k$ de frases '
            f'amostradas e regra de agregação. As frases são sorteadas de forma '
            f'independente; a parte apropriada do vídeo vem só de frases Nenhuma, e frases '
            f'Leve não são usadas (a Tabela~S3 do material suplementar relaxa as duas '
            f'hipóteses). Cada regra e cada $k$ é calibrada para {alarme}\\% de falso alarme '
            f'em vídeos sintéticos inteiramente apropriados da mesma população. {ic_txt}; o '
            f'erro de Monte Carlo de cada ponto ({n_sim:,} vídeos simulados) é de no máximo '
            f'$\\pm${meia_txt} pontos.}}'.replace(f'{n_sim:,}', f'{n_sim:,}'.replace(',', '.'), 1))
        cabecalho = 'Agregação & $k=3$ & $k=5$ & $k=8$ & $k=12$ \\\\'

    linhas = [_abre_tabela(), '\\centering', legenda, '\\label{tab:simulacao}',
              '\\begin{tabular}{lcccc}', '\\toprule', cabecalho]
    for chave, rot_pt, rot_en in pops:
        n_pop = dados['populacoes'][chave]['n']
        rotulo = rot_en if idioma == 'en' else rot_pt
        prefixo = 'Population' if idioma == 'en' else 'População'
        linhas.append('\\midrule')
        linhas.append(f'\\multicolumn{{5}}{{l}}{{\\emph{{{prefixo}: {rotulo} '
                      f'($n = {n_pop}$)}}}} \\\\')
        bloco = dados['resultados'][chave]['deteccao'][variante][f]
        for regra, r_pt, r_en in AGREGACOES:
            celulas = [_celula_pct(bloco[regra][k], idioma) for k in KS]
            linhas.append(' & '.join([r_en if idioma == 'en' else r_pt] + celulas) + ' \\\\')
    linhas += ['\\bottomrule', '\\end{tabular}', _fecha_tabela(), '']
    return '\n'.join(linhas)


def tabela_simulacao_variantes(dados, idioma, regra='media'):
    """
    Variantes da simulação (M5a, M5c) para a regra da Equação do artigo: frases
    i.i.d. com e sem frases Leve na parte apropriada, e segmento tóxico contíguo
    num vídeo ordenado, com posições fixas ou estratificadas. Usa a população OOF
    (a de referência para fixar limiares) ou, na falta dela, o teste.
    """
    if 'resultados' not in dados:
        return None
    pop = 'oof' if 'oof' in dados['resultados'] else 'teste'
    res = dados['resultados'][pop]['deteccao']
    fracoes = [f for f in (dados.get('fracoes_chave') or sorted(res['iid_nenhuma']))]
    L = dados.get('hipoteses', {}).get('bloco_contiguo', {}).get('L_frases_por_video', 200)
    n_pop = dados['populacoes'][pop]['n']
    alarme = dados['falso_alarme_alvo_pct']
    rot_regra = {r: (en if idioma == 'en' else pt) for r, pt, en in AGREGACOES}[regra]

    if idioma == 'en':
        nome_pop = dict((c, en) for c, _, en in POPULACOES)[pop]
        legenda = (
            f'\\caption{{Simulated detection (\\%) under alternative video models, rule '
            f'``{rot_regra}\'\', population: {nome_pop} ($n = {n_pop}$). '
            f'\\emph{{i.i.d.}}: each of the $k$ sampled sentences is severely toxic with '
            f'probability $f$; the appropriate part is drawn from None sentences, or from '
            f'None and Mild pooled (Mild then counts as appropriate content and enters the '
            f'calibration). \\emph{{Contiguous block}}: a video is an ordered sequence of '
            f'$L = {L}$ sentences whose toxic content is one run of $\\mathrm{{round}}(fL)$ '
            f'consecutive Severe sentences starting at a uniformly random position; the '
            f'service reads $k$ positions, either fixed and evenly spaced (for $k = 3$: '
            f'first, middle and last sentence) or one uniformly random position in each of '
            f'$k$ equal strata. The threshold is recalibrated to {alarme}\\% false alarm '
            f'on entirely appropriate videos of the same kind. Confidence intervals are in '
            f'the result file.}}')
        cabecalho = 'Video model & $f$ & $k=3$ & $k=5$ & $k=8$ & $k=12$ \\\\'
    else:
        nome_pop = dict((c, pt) for c, pt, _ in POPULACOES)[pop]
        legenda = (
            f'\\caption{{Detecção simulada (\\%) sob modelos alternativos de vídeo, regra '
            f'``{rot_regra}\'\', população: {nome_pop} ($n = {n_pop}$). \\emph{{i.i.d.}}: '
            f'cada uma das $k$ frases amostradas é severamente tóxica com probabilidade $f$; '
            f'a parte apropriada vem de frases Nenhuma, ou de Nenhuma e Leve reunidas (Leve '
            f'conta então como conteúdo apropriado e entra na calibração). \\emph{{Bloco '
            f'contíguo}}: o vídeo é uma sequência ordenada de $L = {L}$ frases cujo conteúdo '
            f'tóxico é uma sequência de $\\mathrm{{round}}(fL)$ frases Severas consecutivas, '
            f'com início sorteado uniformemente; o serviço lê $k$ posições, fixas e '
            f'igualmente espaçadas (para $k = 3$: primeira, do meio e última frase) ou uma '
            f'posição sorteada em cada um de $k$ estratos iguais. O limiar é recalibrado para '
            f'{alarme}\\% de falso alarme em vídeos inteiramente apropriados do mesmo tipo. '
            f'Os intervalos de confiança estão no arquivo de resultados.}}')
        cabecalho = 'Modelo de vídeo & $f$ & $k=3$ & $k=5$ & $k=8$ & $k=12$ \\\\'

    linhas = [_abre_tabela(), '\\centering', legenda, '\\label{tab:simulacao-variantes}',
              '\\begin{tabular}{llcccc}', '\\toprule', cabecalho]
    for variante, v_pt, v_en in VARIANTES:
        if variante not in res:
            continue
        linhas.append('\\midrule')
        for i, f in enumerate(fracoes):
            rot = (v_en if idioma == 'en' else v_pt) if i == 0 else ''
            celulas = [_celula_pct(res[variante][f][regra][k], idioma, com_ic=False)
                       for k in KS]
            linhas.append(' & '.join([rot, _dec(f'{float(f):.2f}', idioma)] + celulas)
                          + ' \\\\')
    linhas += ['\\bottomrule', '\\end{tabular}', _fecha_tabela(), '']
    return '\n'.join(linhas)


# =============================================================================
# Figuras do estudo de caso
# =============================================================================

def _rotulo_limiar(tau, origem, idioma):
    if origem == 'legado':
        return _dec_txt(f'threshold {tau:.2f}' if idioma == 'en'
                        else f'limiar {tau:.2f}', idioma)
    return _dec_txt(f'a priori threshold τ* = {tau:.3f}' if idioma == 'en'
                    else f'limiar a priori τ* = {tau:.3f}', idioma)


def figura_estratos(dados, idioma, destino_dir=None):
    """Dispersao dos scores por termo, reconstruida de `deslocamentos`.

    Reproduz a figura de `analisar_estratos.py` sem carregar o modelo: os 40
    scores por video ja estao salvos no JSON. O score e o do filtro de toxicidade
    isolado (media das tres frases), nunca o score global com outros filtros.
    """
    import numpy as np

    destino_dir = Path(destino_dir) if destino_dir else destinos()[1]
    tau, origem, _ = limiar_designado(dados)
    rng = np.random.default_rng(3)

    # Ordem dos termos: a mesma de por_termo, que ja vem agrupada por estrato.
    termos = list(dados['por_termo'].keys())
    por_termo = {t: [] for t in termos}
    for d in dados['deslocamentos']:
        if d.get('score') is not None:
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
        rot_a, rot_b = 'A — caregiver query', 'B — child appeal'
        rot_y = 'Video verbal-toxicity score'
    else:
        rot_a, rot_b = 'A — busca do responsável', 'B — apelo infantil'
        rot_y = 'Score do vídeo (só filtro de toxicidade)'

    ax.axhline(tau, color=CINZA_LIMIAR, linestyle='--', linewidth=1.3, zorder=2)
    # Rótulo à esquerda, abaixo da linha: no estrato A quase não há pontos abaixo
    # do limiar (seriam falsos alarmes), e assim não cobre as barras de média do B.
    ax.text(-0.45, tau - 0.015, _rotulo_limiar(tau, origem, idioma),
            fontsize=9, color=CINZA_LIMIAR, ha='left', va='top', zorder=4,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5))
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

    nome_arquivo = 'fig11-estratos.png' if idioma == 'en' else 'fig11-estratos-pt.png'
    caminho = destino_dir / nome_arquivo
    fig.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return caminho


def figura_ordem(dados, idioma, destino_dir=None, com_banda=False):
    """Posicao devolvida pela plataforma (esquerda) contra a posicao apos a
    reordenacao (direita), uma linha por video.

    A figura de estratos mostra QUE os scores separam os dois grupos; esta mostra
    o que essa separacao faz com a lista que o usuario recebe, que e a unidade
    que o servico manipula.

    Cor (m8): duas classes, e so duas, definidas pela mesma regra da tabela de
    limiares -- score >= limiar designado: nao rebaixado (verde); score < limiar
    designado: rebaixado (vermelho escuro). Nao ha faixa intermediaria.

    Com `com_banda`, a ordem da direita e a da reordenacao com banda de
    tolerancia (bloco `banda_tolerancia` do JSON); devolve None se o JSON nao a
    tiver.
    """
    destino_dir = Path(destino_dir) if destino_dir else destinos()[1]
    tau, origem, _ = limiar_designado(dados)

    if com_banda:
        banda = dados.get('banda_tolerancia') or {}
        entradas = banda.get('deslocamentos')
        delta = banda.get('delta')
        if not entradas or delta is None:
            return None
    else:
        entradas = dados['deslocamentos']

    def cor_do_score(s):
        return VERDE if s >= tau else VERMELHO

    termos = list(dados['por_termo'].keys())
    por_termo = {t: [] for t in termos}
    for d in entradas:
        if d.get('score') is not None:
            por_termo[d['termo']].append(d)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    largura, folga = 0.72, 1.55
    n = max(len(v) for v in por_termo.values())

    for i, termo in enumerate(termos):
        x0 = i * folga
        x1 = x0 + largura
        estrato = dados['por_termo'][termo]['estrato']
        cor_est = COR_A if estrato == 'A_responsavel' else COR_B
        for d in por_termo[termo]:
            ax.plot([x0, x1], [d['de'], d['para']], color=cor_do_score(d['score']),
                    alpha=0.85, linewidth=1.3, marker='o', markersize=3.2, zorder=3)
        ax.text(x0 + largura / 2, 1.02, 'A' if estrato == 'A_responsavel' else 'B',
                transform=blended_transform_factory(ax.transData, ax.transAxes),
                ha='center', va='bottom', fontsize=10, fontweight='bold', color=cor_est)

    tau_txt = _dec_txt(f'{tau:.3f}' if origem != 'legado' else f'{tau:.2f}', idioma)
    simbolo = 'τ*' if origem != 'legado' else 'τ'
    if idioma == 'en':
        rot_y = 'Position in the result list'
        titulo = 'Platform ordering (left) $\\rightarrow$ toxicity-score ordering (right)'
        if com_banda:
            titulo += _dec_txt(f', tolerance band δ = {delta:.3f}', idioma)
        faixas = [(VERDE, f'score $\\geq$ {simbolo} = {tau_txt}: not demoted'),
                  (VERMELHO, f'score $<$ {simbolo} = {tau_txt}: demoted')]
    else:
        rot_y = 'Posição na lista de resultados'
        titulo = ('Ordem da plataforma (esquerda) $\\rightarrow$ '
                  'ordem pelo score de toxicidade (direita)')
        if com_banda:
            titulo += _dec_txt(f', banda de tolerância δ = {delta:.3f}', idioma)
        faixas = [(VERDE, f'score $\\geq$ {simbolo} = {tau_txt}: não rebaixado'),
                  (VERMELHO, f'score $<$ {simbolo} = {tau_txt}: rebaixado')]

    ax.set_xlim(-0.35, (len(termos) - 1) * folga + largura + 0.35)
    ax.set_xticks([i * folga + largura / 2 for i in range(len(termos))])
    ax.set_xticklabels([t if len(t) <= 22 else t[:20] + '…' for t in termos], fontsize=9)
    ax.set_ylabel(rot_y, fontsize=10)
    ax.invert_yaxis()
    ax.set_yticks(range(1, n + 1))
    ax.set_ylim(n + 0.6, 0.4)
    ax.grid(axis='y', alpha=0.25, linestyle=':')
    ax.set_axisbelow(True)
    for lado in ('top', 'right'):
        ax.spines[lado].set_visible(False)
    ax.set_title(titulo, fontsize=10, pad=20)

    marcas = [plt.Line2D([], [], color=c, linewidth=2.4, label=r) for c, r in faixas]
    fig.legend(handles=marcas, loc='lower center', ncol=2, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    base = 'fig12-ordem-banda' if com_banda else 'fig12-ordem'
    nome_arquivo = f'{base}.png' if idioma == 'en' else f'{base}-pt.png'
    caminho = destino_dir / nome_arquivo
    fig.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return caminho


# =============================================================================
# Escrita conjunta (usada por analisar_estratos.py e simular_amostragem.py)
# =============================================================================

def escrever_saidas_estratos(dados, idioma, saida=None):
    """Tabelas e figuras do estudo de caso. Devolve a lista de caminhos gravados."""
    tab_dir, fig_dir = destinos(saida)
    gravados = []
    for nome, conteudo in (('tab-estratos.tex', tabela_estratos(dados, idioma)),
                           ('tab-limiares-estudo-caso.tex',
                            tabela_limiares_estudo_caso(dados, idioma)),
                           ('tab-estrato-b.tex', tabela_estrato_b(dados, idioma))):
        (tab_dir / nome).write_text(conteudo, encoding='utf-8')
        gravados.append(tab_dir / nome)
    gravados.append(figura_estratos(dados, idioma, fig_dir))
    gravados.append(figura_ordem(dados, idioma, fig_dir))
    banda = figura_ordem(dados, idioma, fig_dir, com_banda=True)
    if banda:
        gravados.append(banda)
    return gravados


def escrever_saidas_simulacao(dados, idioma, saida=None, f='0.20'):
    """Tabelas da simulação. Devolve a lista de caminhos gravados."""
    tab_dir, _ = destinos(saida)
    gravados = []
    conteudo = tabela_simulacao(dados, idioma, f)
    (tab_dir / 'tab-simulacao.tex').write_text(conteudo, encoding='utf-8')
    gravados.append(tab_dir / 'tab-simulacao.tex')
    variantes = tabela_simulacao_variantes(dados, idioma)
    if variantes:
        (tab_dir / 'tab-simulacao-variantes.tex').write_text(variantes, encoding='utf-8')
        gravados.append(tab_dir / 'tab-simulacao-variantes.tex')
    return gravados


def main():
    global DUAS_COLUNAS

    parser = argparse.ArgumentParser(
        description='Regera as saídas do estudo de caso a partir dos JSON salvos')
    parser.add_argument('--idioma', choices=['pt', 'en'], default='pt')
    parser.add_argument('--fracao-toxica', default='0.20',
                        help='Chave da fração de conteúdo tóxico simulada (padrão 0.20)')
    parser.add_argument('--uma-coluna', dest='duas_colunas', action='store_false',
                        help='Emite table (uma coluna) em vez de table*')
    parser.add_argument('--duas-colunas', dest='duas_colunas', action='store_true', default=True,
                        help='Emite table* em vez de table, para o template do JBCS/JIS')
    parser.add_argument('--resultados', default=None,
                        help='Pasta dos JSON de entrada (padrão: a da rodada)')
    parser.add_argument('--saida', default=None,
                        help='Pasta de saída; grava em <saida>/latex/{tabelas,figuras} '
                             '(padrão: experimento.dir_latex)')
    args = parser.parse_args()

    DUAS_COLUNAS = args.duas_colunas
    pasta = Path(args.resultados) if args.resultados else experimento.dir_resultados()

    estratos, arq_e = _carregar(pasta, 'estratos_*.json')
    simulacao, arq_s = _carregar(pasta, 'simulacao_amostragem_*.json')
    print(f'Fontes:\n  estratos:  {arq_e}\n  simulação: {arq_s}\n')

    for caminho in escrever_saidas_estratos(estratos, args.idioma, args.saida):
        print(f'  {caminho}')
    for caminho in escrever_saidas_simulacao(simulacao, args.idioma, args.saida,
                                             args.fracao_toxica):
        print(f'  {caminho}')


if __name__ == '__main__':
    main()
