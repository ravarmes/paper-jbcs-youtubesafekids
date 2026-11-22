"""
Gera latex/numeros.tex: todo número citado no texto do artigo, como macro.

Regra do repositório: nenhum número entra à mão no .tex. As tabelas saem dos
scripts de cada análise; os números que aparecem na prosa (resumo, seções,
conclusão) saem daqui, lidos dos JSON gravados pela rodada. Mudou o corpus ou
refez o experimento: rode a rodada, rode este script, recompile.

Cada macro tem o prefixo \\ysk e é usada no texto como \\yskFrases{} (as chaves
preservam o espaço seguinte). Um resultado ausente não vira número inventado:
a seção correspondente sai como comentário no .tex e as macros dela ficam
indefinidas, o que quebra a compilação no ponto exato em que o número falta.
Um bloco opcional que falte dentro de uma seção sai como "% NOTA:" no .tex.

Seções: corpus, anotação e partição (dados), e os resultados dos modelos --
busca, modelo final, avaliação, baselines fixas e ajustadas, dispersão,
comparação pareada (Holm), referências que não usam os rótulos (sonda, modelos
prontos, zero-shot, LLM, Detoxify), limiar OOF e calibração, pontos de operação,
simulação, estratos, latência e scores. Cada seção lê o JSON mais recente do
seu padrão na pasta de resultados.

Convenções de formato: F1 com 4 casas; percentuais com 1 casa, sem o símbolo;
diferenças em pontos percentuais (sufixo Pp) em valor absoluto -- quando a
diferença é negativa, a seção ganha uma NOTA; p-valores (sufixo P) já trazem o
sinal de relação e vão em modo matemático: $p \\yskCompSvmP$ -> p = 0.0002;
intervalos com sinal explícito (+0.038); tempos com a unidade no nome (Ms, S).

Uso (de src/; pelo caminho do arquivo, sem -m: app/nlp/evaluation/__init__.py
importa o torch, e este script não precisa dele):
    python app/nlp/evaluation/scripts/gerar_numeros_artigo.py
    python app/nlp/evaluation/scripts/gerar_numeros_artigo.py --resultados <pasta>
    python app/nlp/evaluation/scripts/gerar_numeros_artigo.py --saida <arquivo.tex>
    # com YSK_ESCREVER_LATEX=1 grava em latex/numeros.tex; sem, em results/<tag>/latex/
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd  # noqa: F401

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp import experimento  # noqa: E402

CLASSES = ['Nenhuma', 'Leve', 'Severa']
NOMES_MACRO = {'Nenhuma': 'Nenhuma', 'Leve': 'Leve', 'Severa': 'Severa'}


# ---------------------------------------------------------------------------
# Formatação
# ---------------------------------------------------------------------------

def inteiro(n) -> str:
    """1811 -> '1,811' (separador de milhar do inglês)."""
    return f'{int(round(n)):,}'


def dec(x, casas: int) -> str:
    return f'{float(x):.{casas}f}'


def pct(fracao, casas: int = 1) -> str:
    """Fração (0..1) -> percentual sem o símbolo: 0.7874 -> '78.7'."""
    return f'{float(fracao) * 100:.{casas}f}'


def pp(dif, casas: int = 1) -> str:
    """Diferença de frações em pontos percentuais, sem sinal: 0.053 -> '5.3'."""
    return f'{abs(float(dif)) * 100:.{casas}f}'


def dec_sinal(x, casas: int = 3) -> str:
    """Valor com sinal explícito, para modo matemático: 0.038 -> '+0.038'."""
    return f'{float(x):+.{casas}f}'


def num_curto(x) -> str:
    """Número sem zeros supérfluos: 0.95 -> '0.95', 0.0 -> '0', 1.0 -> '1'."""
    return f'{float(x):g}'


def taxa_aprendizado(lr) -> str:
    """5e-05 -> '5e-5'; 2.5e-05 -> '2.5e-5' (como o texto escreve)."""
    e = math.floor(math.log10(float(lr)))
    return f'{float(lr) / 10 ** e:g}e{e}'


def valor_p(p) -> str:
    """p-valor COM o sinal de relação, para modo matemático: '= 0.40', '= 0.0002',
    '= 1.9\\times10^{-5}'."""
    p = float(p)
    if p <= 0:
        return '< 10^{-300}'
    if p >= 0.1:
        return f'= {p:.2f}'
    if p >= 0.01:
        return f'= {p:.3f}'
    if p >= 0.0001:
        return f'= {p:.4f}'
    e = math.floor(math.log10(p))
    mant = p / 10 ** e
    if round(mant, 1) >= 10:
        mant, e = mant / 10, e + 1
    return f'= {mant:.1f}\\times10^{{{e}}}'


_TEX_ESPECIAIS = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
                  '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
                  '^': r'\textasciicircum{}'}


def tex_escape(s) -> str:
    return ''.join(_TEX_ESPECIAIS.get(ch, ch) for ch in str(s))


_UNIDADES = ['Zero', 'Um', 'Dois', 'Tres', 'Quatro', 'Cinco', 'Seis', 'Sete', 'Oito', 'Nove',
             'Dez', 'Onze', 'Doze', 'Treze', 'Catorze', 'Quinze', 'Dezesseis', 'Dezessete',
             'Dezoito', 'Dezenove']
_DEZENAS = {2: 'Vinte', 3: 'Trinta', 4: 'Quarenta', 5: 'Cinquenta', 6: 'Sessenta',
            7: 'Setenta', 8: 'Oitenta', 9: 'Noventa'}


def por_extenso(n: int) -> str:
    """0..100 por extenso, só letras (nomes de macro): 85 -> 'OitentaCinco'."""
    n = int(n)
    if not 0 <= n <= 100:
        raise ValueError(f'fora de 0..100: {n}')
    if n == 100:
        return 'Cem'
    if n < 20:
        return _UNIDADES[n]
    return _DEZENAS[n // 10] + (_UNIDADES[n % 10] if n % 10 else '')


# ---------------------------------------------------------------------------
# Macros
# ---------------------------------------------------------------------------

class ErroMacro(Exception):
    """Erro de programação (nome inválido ou repetido): não é engolido."""


class Ausente(Exception):
    """Arquivo de resultado não encontrado."""


class Macros:
    """Acumula as macros em ordem, por seção, sem nome repetido."""

    def __init__(self):
        self.secoes = []
        self.nomes = set()
        self.fonte = None        # JSON lido pela seção corrente
        self.tags = {}           # arquivo -> tag de rodada gravada nele
        self.origem = None       # pasta de resultados lida (--resultados)

    def secao(self, titulo: str):
        self.secoes.append({'titulo': titulo, 'itens': [], 'motivo': None, 'notas': [],
                            'fonte': self.fonte})

    def ausente(self, titulo: str, motivo: str):
        self.secoes.append({'titulo': titulo, 'itens': [], 'motivo': motivo, 'notas': [],
                            'fonte': None})

    def nota(self, texto: str):
        self.secoes[-1]['notas'].append(texto)

    def add(self, nome: str, valor: str, nota: str = ''):
        if not nome.isalpha():
            raise ErroMacro(f'nome de macro inválido (só letras): {nome}')
        if nome in self.nomes:
            raise ErroMacro(f'macro repetida: {nome}')
        self.nomes.add(nome)
        self.secoes[-1]['itens'].append((nome, valor, nota))

    def tex(self) -> str:
        linhas = ['% Gerado por app/nlp/evaluation/scripts/gerar_numeros_artigo.py',
                  f'% Rodada: {experimento.tag()}']
        if self.origem is not None:
            linhas.append(f'% Resultados lidos de: {_rel(self.origem)}')
        linhas += ['% NAO EDITAR A MAO: reexecute o script depois de refazer a rodada.', '']
        for s in self.secoes:
            titulo = s['titulo']
            linhas.append(f'% --- {titulo} ' + '-' * max(3, 70 - len(titulo)))
            if s['fonte']:
                linhas.append(f'% Fonte: {_rel(s["fonte"])}')
            if s['motivo']:
                linhas.append(f'% AUSENTE: {s["motivo"]}')
            for n in s['notas']:
                linhas.append(f'% NOTA: {n}')
            for nome, valor, nota in s['itens']:
                comentario = f'  % {nota}' if nota else ''
                linhas.append(f'\\newcommand{{\\ysk{nome}}}{{{valor}}}{comentario}')
            linhas.append('')
        return '\n'.join(linhas)


def _rel(caminho) -> str:
    """Caminho relativo a src/ quando o arquivo está dentro dele; senão, absoluto."""
    try:
        rel = os.path.relpath(str(caminho), project_root)
    except ValueError:  # outra unidade de disco
        rel = '..'
    if rel.startswith('..'):
        return os.path.abspath(str(caminho)).replace(os.sep, '/')
    return rel.replace(os.sep, '/')


def _motivo(e: Exception) -> str:
    if isinstance(e, KeyError):
        return f'chave ausente {e}'
    if isinstance(e, Ausente):
        return str(e)
    return f'{type(e).__name__}: {e}'


def rodar_secao(m: Macros, titulo: str, fn, *args):
    """Roda uma seção; se ela falhar, desfaz o que adicionou e a marca como ausente."""
    m.fonte = None
    n_antes = len(m.secoes)
    try:
        fn(m, *args)
    except ErroMacro:
        raise
    except Exception as e:  # noqa: BLE001 -- a seção ausente não pode derrubar o script
        for s in m.secoes[n_antes:]:
            for nome, _, _ in s['itens']:
                m.nomes.discard(nome)
        del m.secoes[n_antes:]
        prefixo = f'{_rel(m.fonte)}: ' if m.fonte and not isinstance(e, Ausente) else ''
        m.ausente(titulo, prefixo + _motivo(e))
    finally:
        m.fonte = None


@contextmanager
def opcional(m: Macros, rotulo: str):
    """Bloco opcional dentro de uma seção: se faltar chave, vira NOTA e é desfeito."""
    s = m.secoes[-1]
    n = len(s['itens'])
    try:
        yield
    except (KeyError, TypeError, IndexError, ZeroDivisionError, AttributeError,
            StopIteration) as e:
        for nome, _, _ in s['itens'][n:]:
            m.nomes.discard(nome)
        del s['itens'][n:]
        m.nota(f'{rotulo}: não gerado ({_motivo(e)})')


def margem(m: Macros, nome: str, dif, nota: str):
    """Diferença em pp (valor absoluto); diferença negativa ganha NOTA."""
    m.add(nome, pp(dif), nota)
    if float(dif) < 0:
        m.nota(f'{nome}: diferença NEGATIVA ({float(dif):+.4f}); confira o sentido da frase')


def dif_f1(m: Macros, nome: str, dif, nota: str):
    """Diferença de F1 com 4 casas (valor absoluto); diferença negativa ganha NOTA."""
    m.add(nome, dec(abs(float(dif)), 4), nota)
    if float(dif) < 0:
        m.nota(f'{nome}: diferença NEGATIVA ({float(dif):+.4f}); confira o sentido da frase')


def mais_recente(padrao: str):
    arquivos = glob.glob(padrao)
    if not arquivos:
        return None
    return max(arquivos, key=os.path.getmtime)


def ler_json(caminho):
    with open(caminho, encoding='utf-8') as f:
        return json.load(f)


def carregar(m: Macros, res: Path, sub, padrao: str):
    """JSON mais recente de `padrao` em res[/sub]; guarda a fonte para o comentário."""
    pasta = Path(res) / sub if sub else Path(res)
    arq = mais_recente(str(pasta / padrao))
    if arq is None:
        raise Ausente(f'nenhum {padrao} em {_rel(pasta)}')
    m.fonte = arq
    d = ler_json(arq)
    if isinstance(d, dict):
        # a rodada dos DADOS: entradas.experimento_das_entradas quando o script a
        # registra (calibração); senão, o experimento gravado pelo próprio script
        entradas = d.get('entradas') if isinstance(d.get('entradas'), dict) else {}
        metodologia = d.get('methodology') if isinstance(d.get('methodology'), dict) else {}
        exp = (entradas.get('experimento_das_entradas') or d.get('experimento')
               or d.get('experimento_atual') or metodologia.get('experimento'))
        if isinstance(exp, dict) and exp.get('tag'):
            m.tags[_rel(arq)] = exp['tag']
    return d


def registrar_n_teste(ctx: dict, fonte: str, n):
    ctx.setdefault('n_teste', {})[fonte] = int(n)


# ---------------------------------------------------------------------------
# Seções: corpus, anotação e partição
# ---------------------------------------------------------------------------

def secao_corpus(m: Macros, res: Path):
    arq = res / 'corpus' / 'corpus_stats.json'
    if not arq.exists():
        m.ausente('Corpus', f'{arq} não encontrado (rode analisar_corpus)')
        return
    s = ler_json(arq)
    m.secao('Corpus (analisar_corpus.py; corpus inteiro, teste incluído)')
    n = s['total_sentencas']
    m.add('Frases', inteiro(n))
    m.add('Videos', inteiro(s['n_videos']))
    m.add('FrasesDistintas', inteiro(s['textos_distintos']))
    c = s['comprimento_palavras']
    m.add('ComprMedia', dec(c['media'], 1), 'palavras')
    m.add('ComprMediana', inteiro(c['mediana']))
    m.add('ComprPercentilNoventaCinco', inteiro(c['p95']))
    m.add('ComprMax', inteiro(c['max']))
    for cl in CLASSES:
        qtd = s['distribuicao'][cl]
        m.add(f'{cl}N', inteiro(qtd))
        m.add(f'{cl}Pct', dec(s['distribuicao_pct'][cl], 1))
    m.add('ToxicasN', inteiro(n - s['distribuicao']['Nenhuma']))

    k = s['concordancia']
    m.add('UnanimesN', inteiro(k['unanimidade_n']))
    m.add('UnanimesPct', pct(k['unanimidade_pct']))
    m.add('SemMaioriaN', inteiro(k['sem_maioria_n']))
    m.add('SemMaioriaPct', pct(k['sem_maioria_n'] / n))
    m.add('AcordoMedio', pct(k['acordo_par_a_par_medio']))
    m.add('Kappa', dec(k['fleiss_kappa'], 4), 'Fleiss')

    d = s['decidibilidade_lexical']
    m.add('PrevToxicaPct', dec(d['prevalencia_toxica_pct'], 1))
    m.add('ComPalavraoN', inteiro(d['sentencas_com_palavrao']))
    m.add('ComPalavraoPct', dec(d['pct_com_palavrao'], 1))
    m.add('ToxSemPalavraoN', inteiro(d['toxicas_sem_palavrao']))
    m.add('ToxSemPalavraoPct', dec(d['pct_toxicas_sem_palavrao'], 1))
    m.add('NaoToxComPalavraoN', inteiro(d['nao_toxicas_com_palavrao']))
    m.add('NaoToxComPalavraoPct', dec(d['pct_nao_toxicas_com_palavrao'], 1))
    m.add('RegraIngenua', dec(d['regra_ingenua_palavrao_f1_macro'], 2), 'macro F1 binário')
    m.add('RegraIngenuaLonga', dec(d['regra_ingenua_palavrao_f1_macro'], 4))
    m.add('RegraIngenuaPrec', pct(d['regra_ingenua_precisao_toxica']))
    m.add('RegraIngenuaRev', pct(d['regra_ingenua_revocacao_toxica']))
    m.add('SempreToxica', dec(d['referencia_sempre_toxica_f1_macro'], 2))
    m.add('Aleatoria', dec(d['referencia_aleatoria_estratificada_f1_macro'], 2))
    m.add('PalavraPuraN', inteiro(d['sentencas_decidiveis_por_palavra']))
    m.add('PalavraPuraPct', dec(d['pct_decidiveis_por_palavra'], 1))

    ex = s.get('experimentos')
    if ex:
        # Só as frases dos experimentos (v2: sem os grupos de quase-duplicatas)
        ne = ex['total_sentencas']
        m.add('ExpFrases', inteiro(ne), 'frases dos experimentos')
        m.add('QdExcluidas', inteiro(ex['quase_duplicatas_excluidas']), 'fora dos experimentos')
        for cl in CLASSES:
            m.add(f'Exp{cl}N', inteiro(ex['distribuicao'][cl]))
            m.add(f'Exp{cl}Pct', dec(ex['distribuicao_pct'][cl], 1))
        m.add('ExpToxicasN', inteiro(ne - ex['distribuicao']['Nenhuma']))
        if 'n_videos' in ex:
            m.add('ExpVideos', inteiro(ex['n_videos']), 'vídeos com alguma frase nos experimentos')
            m.add('ExpComprMedia', dec(ex['comprimento_palavras']['media'], 1), 'palavras')
            m.add('ExpComprMediana', inteiro(ex['comprimento_palavras']['mediana']))
        ke = ex['concordancia']
        m.add('ExpUnanimesN', inteiro(ke['unanimidade_n']))
        m.add('ExpUnanimesPct', pct(ke['unanimidade_pct']))
        m.add('ExpSemMaioriaN', inteiro(ke['sem_maioria_n']))
        m.add('ExpSemMaioriaPct', pct(ke['sem_maioria_n'] / ne))
        m.add('ExpAcordoMedio', pct(ke['acordo_par_a_par_medio']))
        m.add('ExpKappa', dec(ke['fleiss_kappa'], 4), 'Fleiss')
        de = ex['decidibilidade_lexical']
        m.add('ExpPrevToxicaPct', dec(de['prevalencia_toxica_pct'], 1))
        m.add('ExpComPalavraoN', inteiro(de['sentencas_com_palavrao']))
        m.add('ExpComPalavraoPct', dec(de['pct_com_palavrao'], 1))
        m.add('ExpToxSemPalavraoN', inteiro(de['toxicas_sem_palavrao']))
        m.add('ExpToxSemPalavraoPct', dec(de['pct_toxicas_sem_palavrao'], 1))
        m.add('ExpNaoToxComPalavraoN', inteiro(de['nao_toxicas_com_palavrao']))
        m.add('ExpNaoToxComPalavraoPct', dec(de['pct_nao_toxicas_com_palavrao'], 1))
        m.add('ExpRegraIngenua', dec(de['regra_ingenua_palavrao_f1_macro'], 2), 'macro F1 binário')
        m.add('ExpRegraIngenuaPrec', pct(de['regra_ingenua_precisao_toxica']))
        m.add('ExpRegraIngenuaRev', pct(de['regra_ingenua_revocacao_toxica']))
        m.add('ExpAleatoria', dec(de['referencia_aleatoria_estratificada_f1_macro'], 2))
        m.add('ExpSempreToxica', dec(de['referencia_sempre_toxica_f1_macro'], 2))
        m.add('ExpPalavraPuraN', inteiro(de['sentencas_decidiveis_por_palavra']))
        m.add('ExpPalavraPuraPct', dec(de['pct_decidiveis_por_palavra'], 1))

    p = s['proximidade_treino_teste']
    m.add('AnalogoLimiar', dec(p['limiar'], 1))
    m.add('AnalogoN', inteiro(p['com_analogo_proximo']))
    m.add('AnalogoPct', dec(p['pct_com_analogo_proximo'], 1))
    m.add('SemAnalogoN', inteiro(p['sem_analogo_proximo']))


PAPEIS = {'P1': 'Prof', 'P2': 'Psi', 'A1': 'Aluno'}


def secao_anotacao(m: Macros, res: Path):
    arq = res / 'anotacao' / 'anotacao_stats.json'
    if not arq.exists():
        m.ausente('Anotação', f'{arq} não encontrado (rode analisar_anotacao)')
        return
    s = ler_json(arq)
    m.secao('Anotação (analisar_anotacao.py)')
    for par, d in s['pares'].items():
        a, b = par.split('-')
        nome = PAPEIS[a] + PAPEIS[b]
        m.add(f'Acordo{nome}', pct(d['acordo']))
        m.add(f'Cohen{nome}', dec(d['cohen_kappa'], 4))
    for cl, v in s['fleiss_por_classe'].items():
        m.add(f'Kappa{cl}', dec(v, 3), 'Fleiss por categoria')
    des = s['desacordos']
    m.add('DesacordosN', inteiro(des['n']))
    m.add('DesacNenhumaLeve', inteiro(des['por_par_de_classes']['Nenhuma-Leve']))
    m.add('DesacLeveSevera', inteiro(des['por_par_de_classes']['Leve-Severa']))
    m.add('DesacTres', inteiro(des['por_par_de_classes']['tres_classes']))
    m.add('DesacNenhumaSevera',
          inteiro(des['por_par_de_classes'].get('Nenhuma-Severa', 0)))
    tres = s['tres_rotulos']
    m.add('TresRotulosN', inteiro(len(tres)))
    finais = {t['rotulo_final'] for t in tres}
    m.add('TresRotulosFinal', ' and '.join(sorted(finais)) if finais else '--')

    fpv = s['frases_por_video']
    dist = {int(k): v for k, v in fpv['distribuicao'].items()}
    m.add('VideosSeisFrases', inteiro(dist.get(6, 0)))
    m.add('VideosMaisDeSeis', inteiro(sum(v for k, v in dist.items() if k > 6)))
    m.add('VideosMenosDeSeis', inteiro(sum(v for k, v in dist.items() if k < 6)))
    m.add('FrasesPorVideoMedia', dec(fpv['media'], 1))
    m.add('FrasesPorVideoMax', inteiro(fpv['max']))
    m.add('VideosTresClasses', inteiro(s['videos_com_as_tres_classes']))

    ad = s['animacao_adulta']
    m.add('AdultaVideos', inteiro(ad['videos']))
    m.add('AdultaFrases', inteiro(ad['frases']))
    m.add('AdultaFrasesPct', pct(ad['frases'] / s['n_frases']))
    m.add('AdultaPrevPct', pct(ad['prevalencia_toxica']))
    m.add('DemaisPrevPct', pct(ad['prevalencia_toxica_demais']))
    fr = s['franquias']
    m.add('FranquiasN', inteiro(len([k for k in fr if k != 'outros'])))
    m.add('OutrosVideos', inteiro(fr.get('outros', {}).get('videos', 0)))

    nomes_lote = {'1': 'LoteUm', '2': 'LoteDois'}
    for lote, d in s.get('lotes', {}).items():
        pre = nomes_lote[lote]
        m.add(f'{pre}N', inteiro(d['frases']))
        m.add(f'{pre}Videos', inteiro(d['videos']))
        m.add(f'{pre}VideosNovos', inteiro(d['videos_fora_do_lote1']))
        m.add(f'{pre}VideosAntigos', inteiro(d['videos'] - d['videos_fora_do_lote1']))
        m.add(f'{pre}Kappa', dec(d['fleiss_kappa'], 4))
        m.add(f'{pre}PrevPct', pct(d['prevalencia_toxica']))

    partes = s['partes']
    m.add('ParteInicio', inteiro(partes.get('Início', 0)))
    m.add('ParteMeio', inteiro(partes.get('Meio', 0)))
    m.add('ParteFim', inteiro(partes.get('Fim', 0)))


def secao_particao(m: Macros):
    arq = experimento.ARQ_PARTICAO_VIDEO
    if not arq.exists():
        m.ausente('Partição por vídeo', f'{arq} não encontrado')
        return
    part = experimento.carregar_particao_video()
    m.secao('Partição por vídeo (data/particoes/particao_video.csv)')
    treino = part[part['particao'] == 'treino']
    teste = part[part['particao'] == 'teste']
    m.add('TreinoN', inteiro(len(treino)))
    m.add('TreinoVideos', inteiro(treino['video_id'].nunique()))
    m.add('TesteN', inteiro(len(teste)))
    m.add('TesteVideos', inteiro(teste['video_id'].nunique()))
    # Denominador: frases dos experimentos (na v2, sem as quase-duplicatas excluídas)
    n_exp = len(treino) + len(teste)
    m.add('TestePct', dec(len(teste) / n_exp * 100, 0))
    m.add('TreinoPct', dec(len(treino) / n_exp * 100, 0))
    m.add('VideosAmbosLados',
          inteiro(len(set(treino['video_id']) & set(teste['video_id']))))
    tam = treino.groupby('fold_k5').size()
    m.add('FoldMinN', inteiro(tam.min()))
    m.add('FoldMaxN', inteiro(tam.max()))


# ---------------------------------------------------------------------------
# Seções: resultados dos modelos
# ---------------------------------------------------------------------------

# Nome no JSON -> (sufixo de macro, nome para o texto)
BASELINES_FIXAS = [('SVM (LinearSVC)', 'Svm', 'SVM'),
                   ('Logistic Regression', 'Lr', 'logistic regression'),
                   ('Random Forest', 'Rf', 'random forest'),
                   ('Naive Bayes', 'Nb', 'naive Bayes')]
BASELINES_AJUSTADAS = [('svm', 'SvmAj', 'tuned SVM'),
                       ('lr', 'LrAj', 'tuned logistic regression'),
                       ('cnb', 'CnbAj', 'tuned complement naive Bayes')]
SONDA_COMP = 'Sonda linear (congelado)'
TOKENS_COMP = {SONDA_COMP: 'Sonda',
               'SVM (LinearSVC)': 'Svm', 'Logistic Regression': 'Lr',
               'Random Forest': 'Rf', 'Naive Bayes': 'Nb',
               'SVM (LinearSVC), tuned': 'SvmAj', 'Logistic Regression, tuned': 'LrAj',
               'Complement NB, tuned': 'CnbAj'}
NOMES_TEXTO = {SONDA_COMP: 'linear probe',
               **{n: t for n, _, t in BASELINES_FIXAS},
               'SVM (LinearSVC), tuned': 'tuned SVM',
               'Logistic Regression, tuned': 'tuned logistic regression',
               'Complement NB, tuned': 'tuned complement naive Bayes'}
KS = [(3, 'KTres'), (12, 'KDoze')]


def secao_busca(m: Macros, res: Path, ctx: dict):
    d = carregar(m, res, 'toxicity_gridsearch', 'gridsearch_optimized_search_*.json')
    m.secao('Busca de hiperparâmetros (train_toxicity_gridsearch.py --mode search)')
    s1, s2, s3, best = d['stage1'], d['stage2'], d['stage3'], d['best_config']

    with opcional(m, 'espaço de busca'):
        esp = d['espaco_busca']
        m.add('BuscaEspacoN', inteiro(math.prod(len(v) for v in esp.values())),
              'combinações possíveis (produto dos tamanhos das listas)')
        m.add('BuscaDimensoes', inteiro(sum(1 for v in esp.values() if len(v) > 1)),
              'dimensões com mais de um valor')
    m.add('BuscaConfigs', inteiro(d['n_amostras']), 'configurações sorteadas (estágio 1)')
    m.add('BuscaEstUmFolds', inteiro(s1['folds']))
    m.add('BuscaEstUmEpocas', inteiro(s1['epochs_override']), 'teto de épocas no estágio 1')
    m.add('BuscaEstDoisN', inteiro(len(s2['results'])))
    m.add('BuscaEstDoisFolds', inteiro(s2['folds']))
    m.add('BuscaEstTresN', inteiro(len(s3['results'])))
    m.add('BuscaEstTresSementes', inteiro(len(s3['seeds'])))
    m.add('BuscaEstTresFolds', inteiro(s3['folds']))
    total = (len(s1['results']) * s1['folds'] + len(s2['results']) * s2['folds']
             + len(s3['results']) * len(s3['seeds']) * s3['folds'])
    m.add('BuscaTreinos', inteiro(total),
          'estágio 1 x folds + estágio 2 x folds + estágio 3 x sementes x folds (planejado)')
    falhas = [r for r in s1['results'] + s2['results']
              if r.get('n_folds_ok') is not None
              and r['n_folds_ok'] < (s1['folds'] if r in s1['results'] else s2['folds'])]
    if falhas:
        m.nota(f'{len(falhas)} configurações dos estágios 1-2 com folds que falharam '
               '(BuscaTreinos é o total planejado)')

    c = best['config']
    m.add('VencEpocas', inteiro(c['epochs']))
    m.add('VencBatch', inteiro(c['batch_size']))
    m.add('VencLr', taxa_aprendizado(c['learning_rate']))
    m.add('VencMaxLength', inteiro(c['max_length']), 'tokens')
    m.add('VencWarmupPct', dec(c['warmup_ratio'] * 100, 0))
    m.add('VencWeightDecay', num_curto(c['weight_decay']))
    m.add('VencLlrd', num_curto(c['llrd']))
    m.add('VencPerda', 'focal' if c['loss_type'] == 'focal' else 'cross-entropy')
    m.add('VencSuavizacao', num_curto(c['label_smoothing']), 'label smoothing')
    m.add('VencOversampling', 'yes' if c['oversample'] else 'no')

    m.add('BertCvMedia', dec(best['mean_f1'], 4),
          'F1 macro de CV do vencedor, média entre as sementes do estágio 3')
    m.add('BertCvDpSementes', dec(best['std_f1'], 4),
          'DP entre sementes como gravado na busca (np.std, ddof=0)')
    m.add('BertCvPiorSemente', dec(best['pior_semente'], 4))
    m.add('BertCvSementesN', inteiro(len(best['por_semente'])))
    ctx['bert_cv'] = best['mean_f1']

    lider = max(s2['results'], key=lambda r: r['mean_f1'])
    venc2 = next((r for r in s2['results'] if r['config'] == c), None)
    m.add('VencEraLiderEstDois', 'yes' if venc2 is lider else 'no')
    with opcional(m, 'margem do líder do estágio 2'):
        m.add('EstDoisLiderMargem', dec(lider['mean_f1'] - venc2['mean_f1'], 4),
              'F1 do líder do estágio 2 menos o do vencedor, no estágio 2')

    venc3 = next(r for r in s3['results'] if r['config'] == c)
    lider3 = next((r for r in s3['results'] if r['config'] == lider['config']), None)
    if lider3 is not None and lider3 is not venc3:
        m.add('EstTresLiderDp', dec(lider3['std_f1'], 4), 'DP entre sementes do líder do estágio 2')
        m.add('EstTresLiderMedia', dec(lider3['mean_f1'], 4),
              'F1 médio entre sementes do líder do estágio 2, no estágio 3')
        m.add('EstTresLiderAtras', dec(venc3['mean_f1'] - lider3['mean_f1'], 4),
              'vencedor menos líder do estágio 2, F1 médio entre sementes no estágio 3')
        with opcional(m, 'razão de dispersão'):
            m.add('EstTresRazaoDpLider', dec(lider3['std_f1'] / venc3['std_f1'], 1),
                  'DP entre sementes do líder do estágio 2 / DP do vencedor')
    else:
        m.nota('o vencedor foi o líder do estágio 2 (ou o líder não chegou ao estágio 3): '
               'sem razão de dispersão')
    medias3 = [r['mean_f1'] for r in s3['results']]
    m.add('EstTresAmplitude', dec(max(medias3) - min(medias3), 4),
          'max - min do F1 médio entre os finalistas (estágio 3)')

    outra = 'ce' if c['loss_type'] == 'focal' else 'focal'
    m.add('PerdaOutra', {'ce': 'cross-entropy', 'focal': 'focal'}[outra])
    rivais3 = [r for r in s3['results'] if r['config']['loss_type'] == outra]
    rivais2 = [r for r in s2['results'] if r['config']['loss_type'] == outra]
    if rivais3:
        dif_f1(m, 'PerdaMargem', best['mean_f1'] - max(r['mean_f1'] for r in rivais3),
               'vencedor menos a melhor config. da outra perda, média entre sementes (estágio 3)')
        m.add('PerdaMargemEstagio', '3')
    elif rivais2 and venc2 is not None:
        dif_f1(m, 'PerdaMargem', venc2['mean_f1'] - max(r['mean_f1'] for r in rivais2),
               'vencedor menos a melhor config. da outra perda, no estágio 2 '
               '(nenhum finalista usa a outra perda)')
        m.add('PerdaMargemEstagio', '2')
    else:
        m.nota('nenhuma configuração dos estágios 2-3 usa a outra perda: sem PerdaMargem')

    with opcional(m, 'LLRD no estágio 1'):
        grupos = {}
        for r in s1['results']:
            grupos.setdefault(r['config']['llrd'], []).append(r['mean_f1'])
        medias = {k: sum(v) / len(v) for k, v in grupos.items()}
        m.add('LlrdValores', ', '.join(num_curto(k) for k in sorted(medias)))
        m.add('LlrdAmplitudeEstUm', dec(max(medias.values()) - min(medias.values()), 4),
              'max - min do F1 médio por valor de LLRD (estágio 1)')
        dps = [r['std_f1'] for r in s1['results'] if r.get('std_f1') is not None]
        m.add('EstUmDpMedio', dec(sum(dps) / len(dps), 4),
              'DP médio entre folds no estágio 1 (faixa de ruído da triagem)')


def secao_modelo_final(m: Macros, res: Path, ctx: dict):
    d = carregar(m, res, 'modelo_final', 'modelo_final_*.json')
    m.secao('Modelo final (train_toxicity_gridsearch.py --mode final)')
    r = d['teste_resumo']
    m.add('FinalSementesN', inteiro(len(d['sementes'])))
    m.add('FinalSementeRef', str(d['semente_referencia']))
    m.add('BertTesteMedia', dec(r['f1_macro_media'], 4), 'F1 macro no teste, média entre sementes')
    ctx['bert_teste_media'] = r['f1_macro_media']
    m.add('BertTesteDp', dec(r['f1_macro_dp'], 4), 'DP entre sementes, como gravado')
    m.add('BertTesteMin', dec(r['f1_macro_min'], 4))
    m.add('BertTesteMax', dec(r['f1_macro_max'], 4))
    ref = next(t for t in d['teste_por_semente'] if t['seed'] == d['semente_referencia'])
    m.add('BertTesteRef', dec(ref['f1_macro'], 4), 'F1 macro no teste, semente de referência')
    ctx['bert_teste_final_ref'] = ref['f1_macro']
    with opcional(m, 'F1 das predições out-of-fold'):
        m.add('BertOof', dec(d['oof_f1_macro'], 4), 'F1 macro das predições OOF do treino')
    m.add('FinalNDev', inteiro(d['n_dev']))
    m.add('FinalNTeste', inteiro(d['n_teste']))
    registrar_n_teste(ctx, 'modelo_final', d['n_teste'])


def secao_avaliacao(m: Macros, res: Path, ctx: dict):
    d = carregar(m, res, 'toxicity_evaluation', 'evaluation_metrics_TOX_*.json')
    m.secao('Avaliação no teste (evaluate_toxicity.py; modelo de referência)')
    m.add('BertTesteFUm', dec(d['f1_macro'], 4))
    m.add('BertTesteFUmCurto', dec(d['f1_macro'], 2))
    ctx['bert_teste'] = d['f1_macro']
    ref = ctx.get('bert_teste_final_ref')
    if ref is not None and abs(ref - d['f1_macro']) > 1e-6:
        m.nota(f'ATENÇÃO: F1 da avaliação ({d["f1_macro"]:.4f}) difere do da semente de '
               f'referência no modelo_final ({ref:.4f}): modelos de rodadas diferentes?')
    m.add('BertTesteAcc', pct(d['accuracy']))
    for i, cl in enumerate(CLASSES):
        m.add(f'BertFUm{cl}', pct(d['f1_per_class'][i]))
        m.add(f'BertRec{cl}', pct(d['recall_per_class'][i]))
        m.add(f'BertPrec{cl}', pct(d['precision_per_class'][i]))
        m.add(f'Suporte{cl}', inteiro(d['support_per_class'][i]))
    registrar_n_teste(ctx, 'avaliacao', sum(d['support_per_class']))

    cm = d['confusion_matrix']  # linhas = rótulo verdadeiro (Nenhuma, Leve, Severa)
    m.add('ErrosN', inteiro(sum(cm[i][j] for i in range(3) for j in range(3) if i != j)))
    m.add('ErrosLeveSeveraN', inteiro(cm[1][2] + cm[2][1]),
          'erros na fronteira Leve-Severa (as duas direções)')
    m.add('ErrosNenhumaLeveN', inteiro(cm[0][1] + cm[1][0]),
          'erros entre Nenhuma e Leve (as duas direções)')
    m.add('ErrosNenhumaSeveraN', inteiro(cm[0][2] + cm[2][0]),
          'erros entre Nenhuma e Severa (as duas direções)')
    m.add('LeveComoSevera', inteiro(cm[1][2]))
    m.add('SeveraComoLeve', inteiro(cm[2][1]))
    m.add('SeveraComoNenhuma', inteiro(cm[2][0]))
    m.add('SeveraComoNenhumaPct', pct(cm[2][0] / sum(cm[2])), 'das frases Severa')
    m.add('SeveraErrosN', inteiro(sum(cm[2]) - cm[2][2]))
    m.add('NenhumaComoLeve', inteiro(cm[0][1]))
    m.add('NenhumaComoSevera', inteiro(cm[0][2]))
    m.add('LeveComoNenhuma', inteiro(cm[1][0]))
    # Tarefa binária (tóxico = Leve ou Severa), derivada da mesma matriz: é a tarefa em
    # que o Detoxify é comparado com o limiar padrão.
    vn, fp = cm[0][0], cm[0][1] + cm[0][2]
    fn = cm[1][0] + cm[2][0]
    vp = cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]
    f1_tox = 2 * vp / (2 * vp + fp + fn)
    f1_nao = 2 * vn / (2 * vn + fn + fp)
    m.add('BertBinFUm', dec((f1_tox + f1_nao) / 2, 4),
          'F1 macro binário (tóxico x não tóxico), da matriz de confusão de três classes')
    with opcional(m, 'AUC'):
        m.add('BertAucMacro', dec(d['auc_scores']['macro'], 3))
    with opcional(m, 'frases sem análogo no treino'):
        sem = d['por_similaridade']['sem_analogo']
        m.add('AvalSemAnalogoN', inteiro(sem['n']))
        m.add('BertSemAnalogoFUm', dec(sem['f1_macro'], 4))
        ctx['bert_sem_analogo'] = sem['f1_macro']
    with opcional(m, 'frases com análogo no treino'):
        com = d['por_similaridade']['com_analogo']
        m.add('AvalComAnalogoN', inteiro(com['n']))
        m.add('BertComAnalogoFUm', dec(com['f1_macro'], 4))


def secao_baselines(m: Macros, res: Path, ctx: dict):
    d = carregar(m, res, None, 'baselines_toxicity_*.json')
    m.secao('Baselines clássicas, configuração fixa (train_toxicity_baselines.py)')
    cv = d['cross_validation_results']
    te = d.get('test_results') or {}
    com_cv = [b for b in BASELINES_FIXAS if b[0] in cv]
    if not com_cv:
        raise KeyError('nenhuma das baselines esperadas em cross_validation_results')
    com_te = [b for b in com_cv if b[0] in te]
    if not com_te:
        m.nota('test_results vazio (rodada só de CV?): macros de teste não geradas')
    with opcional(m, 'TF-IDF'):
        mf = re.search(r'max_features=(\d+)', d['methodology']['features'])
        m.add('BaseTfidfMaxFeatures', inteiro(int(mf.group(1))))

    sem_dp = []
    for nome, tok, _ in com_cv:
        m.add(f'{tok}CvFUm', dec(cv[nome]['f1_mean'], 4))
        if 'f1_dp_entre_folds' in cv[nome]:
            m.add(f'{tok}CvDp', dec(cv[nome]['f1_dp_entre_folds'], 4), 'DP amostral entre folds (ddof=1)')
            ctx.setdefault('dp_folds_baselines', []).append(cv[nome]['f1_dp_entre_folds'])
        else:
            sem_dp.append(nome)
    if sem_dp:
        m.nota('formato antigo: sem f1_dp_entre_folds (o f1_std gravado é ddof=0); '
               f'*CvDp não gerado para {", ".join(sem_dp)}')
    melhor_cv = max(com_cv, key=lambda b: cv[b[0]]['f1_mean'])
    m.add('MelhorBaseCvNome', melhor_cv[2])
    m.add('MelhorBaseCvFUm', dec(cv[melhor_cv[0]]['f1_mean'], 4))
    if 'bert_cv' in ctx:
        margem(m, 'MargemCvPp', ctx['bert_cv'] - cv[melhor_cv[0]]['f1_mean'],
               'BERTimbau (CV, média entre sementes) menos a melhor baseline de CV')
    else:
        m.nota('sem a busca: MargemCvPp não gerado')
    if not com_te:
        return

    for nome, tok, _ in com_te:
        m.add(f'{tok}TesteFUm', dec(te[nome]['f1'], 4))
        with opcional(m, f'{nome}: sem análogo'):
            m.add(f'{tok}SemAnalogoFUm', dec(te[nome]['por_similaridade']['sem_analogo']['f1'], 4))
    melhor_te = max(com_te, key=lambda b: te[b[0]]['f1'])
    pior_te = min(com_te, key=lambda b: te[b[0]]['f1'])
    m.add('MelhorBaseTesteNome', melhor_te[2])
    m.add('MelhorBaseTesteFUm', dec(te[melhor_te[0]]['f1'], 4))
    if 'bert_teste' in ctx:
        margem(m, 'MargemTestePp', ctx['bert_teste'] - te[melhor_te[0]]['f1'],
               'BERTimbau (teste, semente de referência) menos a melhor baseline no teste')
        margem(m, 'MargemMaiorPp', ctx['bert_teste'] - te[pior_te[0]]['f1'],
               'contra a baseline mais fraca no teste')
        m.add('MargemMaiorNome', pior_te[2])
    else:
        m.nota('sem a avaliação: MargemTestePp/MargemMaiorPp não gerados')
    if 'bert_teste_media' in ctx:
        margem(m, 'MargemTesteMediaSementesPp', ctx['bert_teste_media'] - te[melhor_te[0]]['f1'],
               'BERTimbau (teste, média entre sementes) menos a melhor baseline no teste')
    with opcional(m, 'melhor baseline sem análogo'):
        sem = {b: te[b[0]]['por_similaridade']['sem_analogo']['f1'] for b in com_te}
        mb = max(sem, key=sem.get)
        m.add('MelhorBaseSemAnalogoNome', mb[2])
        m.add('MelhorBaseSemAnalogoFUm', dec(sem[mb], 4))
        if 'bert_sem_analogo' in ctx:
            margem(m, 'MargemSemAnalogoPp', ctx['bert_sem_analogo'] - sem[mb],
                   'BERTimbau menos a melhor baseline, só frases sem análogo')
    with opcional(m, 'n do teste'):
        registrar_n_teste(ctx, 'baselines', d['methodology']['test_size'])


def secao_baselines_ajustadas(m: Macros, res: Path, ctx: dict):
    d = carregar(m, res, None, 'baselines_ajustados_*.json')
    m.secao('Baselines ajustadas por CV aninhada (train_toxicity_baselines_ajustados.py)')
    r = d['resultados']
    presentes = [b for b in BASELINES_AJUSTADAS if b[0] in r]
    if not presentes:
        raise KeyError('resultados: nenhuma de svm, lr, cnb')
    for chave, tok, _ in presentes:
        b = r[chave]
        m.add(f'{tok}Configs', inteiro(b['n_configuracoes']), 'configurações sorteadas')
        f1 = b['cv_aninhada']['f1_macro']
        m.add(f'{tok}CvFUm', dec(f1['media'], 4), 'CV aninhada')
        m.add(f'{tok}CvDp', dec(f1['dp_entre_folds'], 4), 'DP amostral entre folds (ddof=1)')
        ctx.setdefault('dp_folds_baselines', []).append(f1['dp_entre_folds'])
        m.add(f'{tok}ConfigsDistintas', inteiro(b['cv_aninhada']['n_configs_distintas_escolhidas']),
              'configurações distintas escolhidas nos 5 folds externos')
        m.add(f'{tok}TesteFUm', dec(b['teste']['f1_macro'], 4))
        with opcional(m, f'{chave}: sem análogo'):
            m.add(f'{tok}SemAnalogoFUm', dec(b['teste']['por_similaridade']['sem_analogo']['f1'], 4))
    with opcional(m, 'otimismo de seleção (CV plana menos aninhada)'):
        otim = [r[c]['cv_plana']['melhor_f1_macro']['media'] - r[c]['cv_aninhada']['f1_macro']['media']
                for c, _, _ in presentes]
        m.add('AjOtimismoMin', dec_sinal(min(otim), 4),
              'melhor F1 da CV plana menos o da CV aninhada (otimismo de seleção), menor')
        m.add('AjOtimismoMax', dec_sinal(max(otim), 4), 'idem, maior')
    with opcional(m, 'faixa do DP entre folds das baselines'):
        dps = ctx['dp_folds_baselines']
        m.add('BaseCvDpMin', dec(min(dps), 4), f'menor DP entre folds entre as {len(dps)} baselines')
        m.add('BaseCvDpMax', dec(max(dps), 4), f'maior DP entre folds entre as {len(dps)} baselines')
    melhor_te = max(presentes, key=lambda b: r[b[0]]['teste']['f1_macro'])
    melhor_cv = max(presentes, key=lambda b: r[b[0]]['cv_aninhada']['f1_macro']['media'])
    m.add('MelhorAjTesteNome', melhor_te[2])
    m.add('MelhorAjTesteFUm', dec(r[melhor_te[0]]['teste']['f1_macro'], 4))
    m.add('MelhorAjCvNome', melhor_cv[2])
    m.add('MelhorAjCvFUm', dec(r[melhor_cv[0]]['cv_aninhada']['f1_macro']['media'], 4))
    if 'bert_teste' in ctx:
        margem(m, 'MargemAjTestePp', ctx['bert_teste'] - r[melhor_te[0]]['teste']['f1_macro'],
               'BERTimbau menos a melhor baseline ajustada, no teste')
    if 'bert_cv' in ctx:
        margem(m, 'MargemAjCvPp',
               ctx['bert_cv'] - r[melhor_cv[0]]['cv_aninhada']['f1_macro']['media'],
               'BERTimbau (CV) menos a melhor baseline ajustada (CV aninhada)')
    with opcional(m, 'divisão'):
        m.add('AjNDev', inteiro(d['split']['n_dev']))
        m.add('AjNTeste', inteiro(d['split']['n_teste']))
        registrar_n_teste(ctx, 'baselines_ajustados', d['split']['n_teste'])


def secao_dispersao(m: Macros, res: Path, ctx: dict):
    d = carregar(m, res, None, 'dispersao_*.json')
    m.secao('Dispersão e margens em unidades de DP (analisar_dispersao.py)')
    b = d['bertimbau']['cv']
    m.add('DispBertDpFolds', dec(b['dp_entre_folds'], 4),
          'média entre sementes do DP amostral (ddof=1) entre os 5 folds')
    m.add('DispBertDpSementes', dec(b['dp_entre_sementes'], 4),
          'DP amostral (ddof=1) entre as médias das sementes')
    mg = d['margens']
    m.add('DispMaisForteCvNome', NOMES_TEXTO.get(mg['baseline_mais_forte_cv'],
                                                 tex_escape(mg['baseline_mais_forte_cv'])))
    c = mg['contra_mais_forte_cv']
    m.add('DispMargemCvEmDp', dec(c['cv']['margem_em_dp_folds_combinado'], 1),
          'margem de CV / DP entre folds combinado dos dois modelos')
    m.add('DispMargemCvEmDpBert', dec(c['cv']['margem_em_dp_folds_bertimbau'], 1),
          'margem de CV / DP entre folds do BERTimbau')
    m.add('DispBaseDpFolds', dec(c['cv']['dp_folds_baseline'], 4))
    with opcional(m, 'margem no teste em DP entre sementes'):
        m.add('DispMargemTesteEmDpSementes', dec(c['teste']['margem_em_dp_sementes'], 1))
    with opcional(m, 'diferença pareada por fold'):
        p = c['cv']['pareado_por_fold']
        m.add('DispPareadoMedia', dec(p['media'], 4))
        m.add('DispPareadoDp', dec(p['dp'], 4))
        m.add('DispPareadoFoldsMelhor', inteiro(p['folds_bertimbau_melhor']))
        m.add('DispPareadoNFolds', inteiro(p['n_folds']))


def secao_comparacao(m: Macros, res: Path, ctx: dict):
    d = carregar(m, res, None, 'comparacao_pareada_*.json')
    m.secao('Comparação pareada contra o BERTimbau (comparar_modelos.py)')
    chaves = [k for k in d if k.startswith('comparacoes_vs_')]
    if not chaves:
        raise KeyError('comparacoes_vs_*')
    comps = d[chaves[0]]
    m.add('CompNTeste', inteiro(d['n_teste']))
    registrar_n_teste(ctx, 'comparacao', d['n_teste'])
    m.add('CompNBoot', inteiro(d['n_bootstrap']))
    with opcional(m, 'número de vídeos do teste'):
        m.add('CompNVideos', inteiro(d['n_videos_teste']))
    holm = all('p_holm' in c['bootstrap'] for c in comps.values())
    if not holm:
        m.nota('formato antigo: sem p corrigido nem Holm; *PHolm não gerado e *P usa p_bootstrap')
    for nome, c in comps.items():
        tok = TOKENS_COMP.get(nome)
        if tok is None:
            m.nota(f'modelo sem macro definida: {nome}')
            continue
        bs = c['bootstrap']
        m.add(f'Comp{tok}Dif', dec(bs['diferenca_observada'], 4), 'F1 do BERTimbau menos o do outro')
        m.add(f'Comp{tok}IcBaixo', dec_sinal(bs['ic95_baixo'], 3))
        m.add(f'Comp{tok}IcAlto', dec_sinal(bs['ic95_alto'], 3))
        if 'p_bootstrap_corrigido' in bs:
            m.add(f'Comp{tok}P', valor_p(bs['p_bootstrap_corrigido']),
                  'bootstrap pareado por frase, p corrigido; inclui "="')
        elif bs['p_bootstrap'] > 0:
            m.add(f'Comp{tok}P', valor_p(bs['p_bootstrap']), 'bootstrap pareado por frase; inclui "="')
        else:  # nenhuma reamostragem cruzou o zero: só se sabe que p < 2/(B+1)
            m.add(f'Comp{tok}P', f'< {2 / (d["n_bootstrap"] + 1):.4f}',
                  'nenhuma reamostragem cruzou o zero; limite 2/(B+1)')
        if 'p_holm' in bs:
            m.add(f'Comp{tok}PHolm', valor_p(bs['p_holm']))
        mc = c['mcnemar']
        m.add(f'Comp{tok}McNemarP', valor_p(mc['p_valor']))
        m.add(f'Comp{tok}SoBert', inteiro(mc['so_a']), 'frases que só o BERTimbau acerta')
        m.add(f'Comp{tok}SoOutro', inteiro(mc['so_b']), 'frases que só o outro acerta')
        bv = c.get('bootstrap_por_video')
        if bv:
            m.add(f'Comp{tok}VideoIcBaixo', dec_sinal(bv['ic95_baixo'], 3))
            m.add(f'Comp{tok}VideoIcAlto', dec_sinal(bv['ic95_alto'], 3))
            if 'p_holm' in bv:
                m.add(f'Comp{tok}VideoPHolm', valor_p(bv['p_holm']))

    bases = [n for n in comps if n in TOKENS_COMP and n != SONDA_COMP]
    if bases:
        menor = min(bases, key=lambda n: comps[n]['bootstrap']['diferenca_observada'])
        maior = max(bases, key=lambda n: comps[n]['bootstrap']['diferenca_observada'])
        m.add('CompMenorMargemNome', NOMES_TEXTO[menor])
        m.add('CompMenorMargemDif', dec(comps[menor]['bootstrap']['diferenca_observada'], 4))
        m.add('CompMenorMargemMcNemarP', valor_p(comps[menor]['mcnemar']['p_valor']))
        if holm:
            m.add('CompMenorMargemPHolm', valor_p(comps[menor]['bootstrap']['p_holm']))
        m.add('CompMaiorMargemNome', NOMES_TEXTO[maior])
        m.add('CompMaiorMargemDif', dec(comps[maior]['bootstrap']['diferenca_observada'], 4))
        m.add('CompMaiorMargemPp', pp(comps[maior]['bootstrap']['diferenca_observada']))
        m.add('CompNBaselines', inteiro(len(bases)), 'baselines comparadas (fixas e ajustadas)')
        if holm:
            m.add('CompNSignifHolm',
                  inteiro(sum(comps[n]['bootstrap']['significativo_holm'] for n in bases)),
                  'baselines (fixas e ajustadas) separadas do BERTimbau após Holm (bootstrap por frase)')
            if all('significativo_holm' in (comps[n].get('bootstrap_por_video') or {})
                   for n in bases):
                m.add('CompNSignifHolmVideo',
                      inteiro(sum(comps[n]['bootstrap_por_video']['significativo_holm']
                                  for n in bases)),
                      'idem, bootstrap por vídeo')
    rob = d.get('robustez_sementes_bertimbau')
    if rob:
        with opcional(m, 'robustez à semente'):
            resumo = rob['resumo']
            n_sem = {r['n_sementes'] for r in resumo.values()}
            m.add('CompSementesN', inteiro(max(n_sem)))
            for nome, r in resumo.items():
                tok = TOKENS_COMP.get(nome)
                if tok is None:
                    continue
                m.add(f'Comp{tok}DifMinSementes', dec(r['diferenca_min'], 4))
                m.add(f'Comp{tok}DifMaxSementes', dec(r['diferenca_max'], 4))
                m.add(f'Comp{tok}SementesSignif',
                      inteiro(r['n_sementes_significativo_holm_frases']),
                      'sementes em que a diferença é significativa após Holm (frases)')
    ref = d['f1_macro_teste'].get('BERTimbau (ajuste fino)')
    if ref is not None and 'bert_teste' in ctx and abs(ref - ctx['bert_teste']) > 1e-6:
        m.nota(f'ATENÇÃO: F1 do BERTimbau na comparação ({ref:.4f}) difere do da avaliação '
               f'({ctx["bert_teste"]:.4f})')


def secao_sem_finetuning(m: Macros, res: Path, ctx: dict):
    d = carregar(m, res, None, 'sem_finetuning_*.json')
    m.secao('Referências sem ajuste fino: sonda, modelos prontos, zero-shot '
            '(train_toxicity_sem_finetuning.py)')
    por_nome = {r['modelo']: r for r in d['resultados']}
    with opcional(m, 'n do teste'):
        m.add('SemFtNTeste', inteiro(d['n_teste']))
        registrar_n_teste(ctx, 'sem_finetuning', d['n_teste'])
    with opcional(m, 'sonda linear'):
        s = por_nome['Sonda linear (BERTimbau congelado)']
        m.add('SondaFUm', dec(s['f1_macro'], 4))
        m.add('SondaCvFUm', dec(s['cv_f1_mean'], 4))
        m.add('SondaCvDp', dec(s['cv_f1_std'], 4), 'np.std entre folds (ddof=0), como gravado')
        if 'bert_teste' in ctx:
            margem(m, 'SondaGapPp', ctx['bert_teste'] - s['f1_macro'],
                   'BERTimbau ajustado menos a sonda, no teste')
    with opcional(m, 'modelos prontos'):
        for nome, tok in (('Modelos prontos compostos (HateBR + alvo)', 'Prontos'),
                          ('HateBR isolado (tóxico vs. não-tóxico)', 'HateBr'),
                          ('Classificador de alvo isolado (Leve vs. Severa)', 'Alvo')):
            f1 = por_nome[nome]['f1_macro']
            m.add(f'{tok}FUm', dec(f1, 4))
            m.add(f'{tok}FUmCurto', dec(f1, 2))
    with opcional(m, 'zero-shot: moldes'):
        zs = [r for r in d['resultados'] if 'molde' in r]
        if not zs:
            raise KeyError('resultados com "molde"')
        f1s = [r['f1_macro'] for r in zs]
        m.add('ZeroShotMoldesN', inteiro(len(zs)))
        m.add('ZeroShotTesteMin', dec(min(f1s), 4), 'pior molde no teste (registro)')
        m.add('ZeroShotTesteMax', dec(max(f1s), 4), 'melhor molde no teste (registro, não é a referência)')
    with opcional(m, 'zero-shot: molde escolhido no treino'):
        sel = d['zero_shot_selecao']
        ftr = sel['f1_treino_por_molde']
        esc = sel['molde_escolhido_no_treino']
        m.add('ZeroShotFUm', dec(sel['f1_teste_do_escolhido'], 4), 'molde escolhido no treino, F1 no teste')
        m.add('ZeroShotFUmCurto', dec(sel['f1_teste_do_escolhido'], 2))
        m.add('ZeroShotTreinoFUm', dec(ftr[esc], 4), 'F1 do molde escolhido, no treino')
        m.add('ZeroShotTreinoMin', dec(min(ftr.values()), 4))
        m.add('ZeroShotTreinoMax', dec(max(ftr.values()), 4))
        m.add('ZeroShotTreinoN', inteiro(sel['n_treino']))
        fte = sel['f1_teste_por_molde_so_registro']
        m.add('ZeroShotEscolhidoEraMelhorTeste',
              'yes' if max(fte, key=fte.get) == esc else 'no')


def secao_llm(m: Macros, res: Path, ctx: dict):
    d = carregar(m, res, None, 'llm_baselines_toxicity_*.json')
    m.secao('LLM generativo, few-shot (train_toxicity_llm_baselines.py)')
    nomes = list(d['results'])
    nome = 'llama3.1:8b' if 'llama3.1:8b' in nomes else nomes[0]
    if len(nomes) > 1:
        m.nota(f'mais de um modelo no JSON ({", ".join(nomes)}): macros de {nome}')
    r = d['results'][nome]
    m.add('LlmNome', tex_escape(nome))
    m.add('LlmNAmostras', inteiro(r['n_amostras']))
    registrar_n_teste(ctx, 'llm', r['n_amostras'])
    m.add('LlmFUm', dec(r['f1'], 4))
    m.add('LlmFUmCurto', dec(r['f1'], 2))
    m.add('LlmAcc', pct(r['accuracy']))
    cm = r['confusion_matrix']
    n0, n2 = sum(cm[0]), sum(cm[2])
    m.add('LlmRecSevera', pct(cm[2][2] / n2))
    m.add('LlmSeverasPerdidas', inteiro(n2 - cm[2][2]), 'Severas não preditas Severa')
    m.add('LlmFaToxico', pct((n0 - cm[0][0]) / n0), 'Nenhuma predita Leve ou Severa')
    m.add('LlmFaSevera', pct(cm[0][2] / n0), 'Nenhuma predita Severa')
    m.add('LlmNenhumaComoLeve', inteiro(cm[0][1]))
    m.add('LlmNenhumaN', inteiro(n0))
    if 'bert_teste' in ctx:
        margem(m, 'LlmGapPp', ctx['bert_teste'] - r['f1'], 'BERTimbau ajustado menos o LLM, no teste')
    with opcional(m, 'tempo por frase'):
        m.add('LlmPorFraseS', dec(r['segundos'] / r['n_amostras'], 2),
              'tempo total da avaliação / frases (laço sequencial; ver também Latência)')
    with opcional(m, 'saídas fora do formato'):
        s = r['saidas']
        m.add('LlmForaFormato', inteiro(s['n_fora_do_formato']))
        m.add('LlmForaFormatoPct', pct(s['n_fora_do_formato'] / r['n_amostras']))
        m.add('LlmFallbackNenhuma', inteiro(s['n_fallback_nenhuma']))
    with opcional(m, 'latência por requisição'):
        lat = r['latencia_requisicao_s']
        m.add('LlmReqMedianaS', dec(lat['mediana'], 2))
        m.add('LlmReqPNoventaCincoS', dec(lat['p95'], 2))
    with opcional(m, 'sem análogo'):
        m.add('LlmSemAnalogoFUm', dec(r['por_similaridade']['sem_analogo']['f1'], 4))
    with opcional(m, 'exemplos por classe'):
        m.add('LlmFewShotPorClasse', inteiro(d['methodology']['fewshot_per_class']))


def secao_detoxify(m: Macros, res: Path, ctx: dict):
    d = carregar(m, res, 'detoxify', 'detoxify_*.json')
    m.secao('Detoxify multilíngue, sem ajuste fino (avaliar_detoxify.py)')
    r = d['resultados']
    for chave, tok in (('binaria_padrao', 'DetoxBinPadrao'), ('binaria_treino', 'DetoxBinTreino'),
                       ('tres_limiares', 'DetoxTresLimiares'),
                       ('tres_logistica', 'DetoxTresLogistica')):
        with opcional(m, chave):
            m.add(f'{tok}FUm', dec(r[chave]['teste']['f1_macro'], 4))
    with opcional(m, 'limiar escolhido no treino'):
        m.add('DetoxLimiarTreino', dec(r['binaria_treino']['limiar'], 3))
    m.add('DetoxNTeste', inteiro(d['n_teste']))
    registrar_n_teste(ctx, 'detoxify', d['n_teste'])


def _taxas(m: Macros, prefixo: str, t: dict, ic: bool = True):
    m.add(f'{prefixo}RecSevera', pct(t['recall_severa']))
    m.add(f'{prefixo}SeverasNaoSinal', inteiro(t['severas_nao_sinalizadas']))
    m.add(f'{prefixo}SeverasComoNenhuma', inteiro(t['severas_como_nenhuma']))
    m.add(f'{prefixo}FaSevera', pct(t['fa_severa']), 'Nenhuma sinalizada como Severa')
    m.add(f'{prefixo}FaToxico', pct(t['fa_toxico']), 'Nenhuma sinalizada em qualquer grau')
    m.add(f'{prefixo}PrecSevera', pct(t['precisao_severa']))
    if ic and 'ic95_wilson' in t:
        w = t['ic95_wilson']
        for chave, nome in (('recall_severa', 'RecSevera'), ('fa_severa', 'FaSevera'),
                            ('fa_toxico', 'FaToxico')):
            m.add(f'{prefixo}{nome}IcBaixo', pct(w[chave][0]), 'IC 95% de Wilson')
            m.add(f'{prefixo}{nome}IcAlto', pct(w[chave][1]))


def secao_calibracao(m: Macros, res: Path, ctx: dict):
    d = carregar(m, res, None, 'calibracao_limiar_*.json')
    m.secao('Limiar escolhido nas OOF e calibração (analisar_calibracao_limiar.py)')
    ent = d.get('entradas') or {}
    if ent.get('mesma_rodada_que_a_atual') is False:
        m.nota('ATENÇÃO: entradas de outra rodada '
               f'({(ent.get("experimento_das_entradas") or {}).get("tag")})')
    m.add('CalibNOof', inteiro(d['n']['oof']))
    m.add('CalibNTeste', inteiro(d['n']['teste']))
    registrar_n_teste(ctx, 'calibracao', d['n']['teste'])

    lo = d['limiar_oof']
    arg = lo['argmax']
    _taxas(m, 'Argmax', arg['teste'])
    _taxas(m, 'ArgmaxOof', arg['oof'], ic=False)

    a = lo['regras']['a_orcamento_fa']
    m.add('TauA', dec(a['tau'], 2),
          'regra a (declarada antes): menor tau com FA Severa OOF até o orçamento acima do argmax')
    m.add('TauAOrcamentoPp', dec(a['orcamento_pp'], 1))
    m.add('TauATetoFaOof', pct(a['teto_fa_oof']))
    _taxas(m, 'TauA', a['teste'])
    _taxas(m, 'TauAOof', a['oof'], ic=False)
    margem(m, 'TauAGanhoRecPp', a['teste']['recall_severa'] - arg['teste']['recall_severa'],
           'recall Severa: tau da regra a menos argmax, no teste')
    margem(m, 'TauACustoFaSeveraPp', a['teste']['fa_severa'] - arg['teste']['fa_severa'],
           'FA Severa: tau da regra a menos argmax, no teste')
    margem(m, 'TauACustoFaToxicoPp', a['teste']['fa_toxico'] - arg['teste']['fa_toxico'],
           'FA tóxico: tau da regra a menos argmax, no teste')
    with opcional(m, 'regra a entre sementes'):
        rs = a['teste_resumo_sementes']
        m.add('TauARecSeveraMinSementes', pct(rs['recall_severa']['min']))
        m.add('TauARecSeveraMaxSementes', pct(rs['recall_severa']['max']))
        m.add('TauAFaSeveraMinSementes', pct(rs['fa_severa']['min']))
        m.add('TauAFaSeveraMaxSementes', pct(rs['fa_severa']['max']))
    with opcional(m, 'argmax entre sementes'):
        rs = arg['teste_resumo_sementes']
        m.add('ArgmaxRecSeveraMinSementes', pct(rs['recall_severa']['min']))
        m.add('ArgmaxRecSeveraMaxSementes', pct(rs['recall_severa']['max']))

    with opcional(m, 'regra b (F2 máximo)'):
        b = lo['regras']['b_max_f2']
        m.add('TauB', dec(b['tau'], 2), 'regra b: tau que maximiza o F2 da Severa nas OOF')
        _taxas(m, 'TauB', b['teste'])
        _taxas(m, 'TauBOof', b['oof'], ic=False)

    with opcional(m, 'exploratório: tau escolhido no teste'):
        ex = d['exploratorio_escolha_no_teste']
        m.add('ExplTauA', dec(ex['a_orcamento_fa']['tau'], 2), 'EXPLORATÓRIO: escolhido no teste')
        _taxas(m, 'ExplTauA', ex['a_orcamento_fa']['teste'], ic=False)
        m.add('ExplTauB', dec(ex['b_max_f2']['tau'], 2), 'EXPLORATÓRIO: escolhido no teste')
        _taxas(m, 'ExplTauB', ex['b_max_f2']['teste'], ic=False)

    with opcional(m, 'Severas preditas Nenhuma'):
        s = d['severas_preditas_nenhuma']
        m.add('SevNenhumaN', inteiro(s['n']))
        m.add('SevNenhumaDe', inteiro(s['n_severa_teste']))
        m.add('SevNenhumaPct', pct(s['n'] / s['n_severa_teste']))
        m.add('SevNenhumaSinalizadasTau', inteiro(sum(f['sinalizada_pelo_tau_oof'] for f in s['frases'])),
              'quantas delas o tau OOF sinaliza')
        if s['frases']:
            m.add('SevNenhumaScoreMediano', dec(s['score_mediano_perdidas'], 3))
            m.add('SevNenhumaScoreMin', dec(min(f['score'] for f in s['frases']), 3))
        m.add('NenhumaCorretasScoreMediano', dec(s['score_mediano_nenhuma_corretas'], 3))

    with opcional(m, 'calibração'):
        c = d['calibracao']
        m.add('CalibBins', inteiro(c['n_bins']))
        m.add('Temperatura', dec(c['temperatura']['T'], 2), 'ajustada nas OOF')
        for fase, tag in (('antes', 'Antes'), ('depois', 'Depois')):
            t = c[fase]['teste']
            m.add(f'Ece{tag}', dec(t['ece_top_label'], 3), f'teste, {fase} da temperatura')
            m.add(f'EceSevera{tag}', dec(t['ece_severa'], 3))
            m.add(f'Brier{tag}', dec(t['brier_multiclasse'], 3))
            m.add(f'Nll{tag}', dec(t['nll'], 3))
        m.add('EceOofAntes', dec(c['antes']['oof']['ece_top_label'], 3))
        e = c['efeito_no_score']['teste']
        m.add('CalibScoreSpearman', dec(e['spearman_antes_depois'], 4))
        m.add('CalibScoreDeltaMedio', dec(e['delta_abs_medio'], 3))
        m.add('CalibScoreDeltaMax', dec(e['delta_abs_max'], 3))
        m.add('CalibArgmaxPreservado', 'yes' if e['argmax_preservado'] else 'no')

    with opcional(m, 'sensibilidade aos pesos do score'):
        sp = d['sensibilidade_pesos']
        m.add('PesosLeveTestados', ', '.join(num_curto(x) for x in sp['pesos_leve']))
        m.add('PesosSpearmanMin', dec(min(sp['teste']['spearman_entre_pesos'].values()), 3),
              'menor Spearman entre as ordens produzidas por pesos diferentes da Leve')


def secao_pontos_operacao(m: Macros, res: Path, ctx: dict):
    d = carregar(m, res, None, 'pontos_operacao_*.json')
    m.secao('Pontos de operação com tau escolhido no teste '
            '(analisar_pontos_operacao.py; EXPLORATÓRIO)')
    linhas = [p for p in d['pontos_operacao'] if str(p['modelo']).startswith('BERTimbau (tau')]
    if not linhas:
        raise KeyError("pontos_operacao: linha 'BERTimbau (tau = ...)'")
    p = linhas[0]
    tau = p['tau'] if 'tau' in p else float(p['modelo'].split('=')[1].strip(' )'))
    m.add('PoTau', num_curto(tau), 'EXPLORATÓRIO: escolhido no teste')
    m.add('PoTauRecSevera', pct(p['recall_severa']))
    m.add('PoTauSeverasPerdidas', inteiro(p['severas_perdidas']))
    m.add('PoTauFaSevera', pct(p['fa_severa']))
    m.add('PoTauFaToxico', pct(p['fa_toxico']))
    with opcional(m, 'n do teste'):
        registrar_n_teste(ctx, 'pontos_operacao', d['n_teste'])


REGRAS_SIM = [('media', 'Media'), ('minimo', 'Minimo'), ('min_ponderado', 'MinPond'),
              ('duas_piores', 'DuasPiores')]
VARIANTES_SIM = [('bloco_fixo', 'BlocoFixo'), ('bloco_estratificado', 'BlocoEstrat'),
                 ('iid_nenhuma_leve', 'NenhumaLeve')]


def _bloco_simulacao(m: Macros, prefixo: str, det: dict, fk: str):
    for regra, rt in REGRAS_SIM:
        for k, kt in KS:
            c = det[fk][regra][str(k)]
            m.add(f'{prefixo}{rt}{kt}', pct(c['deteccao']), f'detecção (%), f = {fk}')
            ic = c.get('ic95_bootstrap_populacao')
            if ic:
                m.add(f'{prefixo}{rt}{kt}IcBaixo', pct(ic[0]), 'IC 95% bootstrap da população')
                m.add(f'{prefixo}{rt}{kt}IcAlto', pct(ic[1]))
    k3 = [det[fk][r][str(3)]['deteccao'] for r, _ in REGRAS_SIM]
    m.add(f'{prefixo}AmplitudeKTresPp', pp(max(k3) - min(k3)),
          'max - min da detecção entre as quatro regras, k = 3')


def secao_simulacao(m: Macros, res: Path, ctx: dict):
    d = carregar(m, res, None, 'simulacao_amostragem_*.json')
    m.secao('Simulação de amostragem (simular_amostragem.py)')
    resultados = d['resultados']  # formato antigo (sem populações) cai aqui
    la = d['limiar_a_priori']
    pop = la['populacao']
    fk = f'{float(d["f_tabela"]):.2f}'
    m.add('SimN', inteiro(d['n_simulacoes']), 'vídeos simulados por célula')
    m.add('SimFaAlvoPct', inteiro(d['falso_alarme_alvo_pct']))
    m.add('SimF', fk, 'fração de frases Severa no vídeo de risco (tabela)')
    m.add('SimFPct', dec(float(fk) * 100, 0))
    m.add('SimPopulacao', 'out-of-fold' if pop == 'oof' else 'test')
    m.add('SimPopN', inteiro(d['populacoes'][pop]['n']))
    with opcional(m, 'população de teste'):
        m.add('SimPopTesteN', inteiro(d['populacoes']['teste']['n']))
    if la.get('aviso'):
        m.nota(f'limiar a priori: {la["aviso"]}')
    m.add('LimiarAPriori', dec(la['valor'], 3),
          f'percentil {la["falso_alarme_alvo_pct"]} do score de vídeos só Nenhuma, k = {la["k"]}, '
          f'regra {la["regra"]}, população {pop}')
    m.add('LimiarAPrioriK', inteiro(la['k']))
    m.add('LimiarAPrioriFaVerif', pct(la['falso_alarme_verificado']))
    m.add('LimiarAPrioriIcBaixo', dec(la['ic95_bootstrap_populacao'][0], 3))
    m.add('LimiarAPrioriIcAlto', dec(la['ic95_bootstrap_populacao'][1], 3))
    with opcional(m, 'banda de tolerância'):
        b = d['banda_tolerancia']
        m.add('BandaDelta', dec(b['delta'], 3),
              f'percentil {b["percentil"]} de |S1 - S2| entre vídeos só Nenhuma')
        m.add('BandaPercentil', inteiro(b['percentil']))
        m.add('BandaIcBaixo', dec(b['ic95_bootstrap_populacao'][0], 3))
        m.add('BandaIcAlto', dec(b['ic95_bootstrap_populacao'][1], 3))
        if b.get('aviso'):
            m.nota(f'banda: {b["aviso"]}')

    _bloco_simulacao(m, 'Sim', resultados[pop]['deteccao']['iid_nenhuma'], fk)
    if pop == 'oof' and 'teste' in resultados:
        with opcional(m, 'população de teste'):
            _bloco_simulacao(m, 'SimTeste', resultados['teste']['deteccao']['iid_nenhuma'], fk)
    for var, vt in VARIANTES_SIM:
        with opcional(m, f'variante {var}'):
            det = resultados[pop]['deteccao'][var]
            for regra, rt in (('media', 'Media'), ('min_ponderado', 'MinPond')):
                for k, kt in KS:
                    m.add(f'Sim{vt}{rt}{kt}', pct(det[fk][regra][str(k)]['deteccao']))


def secao_estratos(m: Macros, res: Path, ctx: dict):
    d = carregar(m, res, None, 'estratos_*.json')
    m.secao('Estudo de caso: estratos de busca (analisar_estratos.py)')
    if 'STUB' in str(d.get('modelo', '')).upper():
        m.nota('ATENÇÃO: scores de STUB (teste de formato), não do modelo da rodada')
    with opcional(m, 'totais'):
        m.add('EstNVideos', inteiro(d['n_videos']))
        m.add('EstNBuscas', inteiro(d['n_buscas']))
        m.add('EstSemTranscricao', inteiro(d['n_videos_sem_transcricao']))
    A, B = d['estrato_A_responsavel'], d['estrato_B_apelo_infantil']
    for X, e in ((A, 'A'), (B, 'B')):
        m.add(f'Est{e}NVideos', inteiro(X['n']))
        m.add(f'Est{e}Media', dec(X['media'], 3))
        m.add(f'Est{e}Min', dec(X['min'], 3))
        m.add(f'Est{e}Max', dec(X['max'], 3))
        m.add(f'Est{e}Amplitude', dec(X['max'] - X['min'], 3))
    s = d['separacao_media']
    m.add('EstDifMedias', dec(s['diferenca_medias'], 3))
    u = float(s['mannwhitney_u'])
    m.add('EstU', inteiro(u) if u.is_integer() else dec(u, 1), 'Mann-Whitney')
    m.add('EstP', valor_p(s['p']), 'inclui "="; modo matemático')
    m.add('EstAuc', dec(s['auc'], 3), 'AUC A x B, score = média das frases')
    m.add('EstAucIcBaixo', dec(s['auc_ic95'][0], 3))
    m.add('EstAucIcAlto', dec(s['auc_ic95'][1], 3))
    with opcional(m, 'menor score do estudo'):
        vs = d['videos']
        m.add('EstScoreMin', dec(min(v['score_media'] for v in vs), 3))
    with opcional(m, 'separação pelo mínimo'):
        m.add('EstAucMinimo', dec(d['separacao_minimo']['auc'], 3))
    with opcional(m, 'limiar designado (a priori)'):
        ld = d['limiar_designado']
        r = ld['resultado']
        m.add('EstLimiar', dec(ld['valor'], 3), f'origem: {ld.get("origem")}')
        m.add('EstDetB', inteiro(r['detectados_b']))
        m.add('EstTotB', inteiro(r['total_b']))
        m.add('EstFalsosA', inteiro(r['falsos_a']))
        m.add('EstTotA', inteiro(r['total_a']))
        if ld.get('aviso'):
            m.nota(f'limiar designado: {ld["aviso"]}')
    with opcional(m, 'limiares exploratórios'):
        for linha in d['deteccao_por_limiar']:
            if not linha.get('exploratorio', not linha.get('a_priori', False)):
                continue
            nome = 'EstLimiar' + por_extenso(round(float(linha['limiar']) * 100))
            m.add(f'{nome}DetB', inteiro(linha['detectados_b']),
                  f'limiar {linha["limiar"]} (EXPLORATÓRIO)')
            m.add(f'{nome}FalsosA', inteiro(linha['falsos_a']))
    for sub, tok in (('sem_treino', 'EstSemTreino'), ('sem_sobreposicao', 'EstSemSobrep')):
        with opcional(m, f'subconjunto {sub}'):
            x = d['metricas_por_subconjunto'][sub]
            m.add(f'{tok}NA', inteiro(x['n_a']))
            m.add(f'{tok}NB', inteiro(x['n_b']))
            m.add(f'{tok}Auc', dec(x['auc'], 3))
            m.add(f'{tok}AucIcBaixo', dec(x['auc_ic95'][0], 3))
            m.add(f'{tok}AucIcAlto', dec(x['auc_ic95'][1], 3))
            ld = x['limiar_designado']
            m.add(f'{tok}DetB', inteiro(ld['detectados_b']))
            m.add(f'{tok}FalsosA', inteiro(ld['falsos_a']))
    with opcional(m, 'banda de tolerância'):
        bt = d['banda_tolerancia']
        m.add('EstBandaDelta', dec(bt['delta'], 3))
        for modo, mt in (('sem_banda', 'SemBanda'), ('com_banda', 'ComBanda')):
            for est, e in (('A_responsavel', 'A'), ('B_apelo_infantil', 'B')):
                x = bt[modo][est]
                m.add(f'Est{mt}Desloc{e}', dec(x['deslocamento_medio_abs'], 1),
                      'deslocamento médio |posição| na lista')
                m.add(f'Est{mt}Alteradas{e}', inteiro(x['n_posicoes_alteradas']))
                m.add(f'Est{mt}TopTres{e}', dec(x['sobreposicao_top3_media'], 2),
                      'sobreposição média do Top-3')
                m.add(f'Est{mt}NovoPrimeiro{e}', inteiro(x['buscas_com_novo_primeiro']))
                rb = x['rebaixados_no_limiar']
                m.add(f'Est{mt}Rebaixados{e}', inteiro(rb['n']))
                m.add(f'Est{mt}RebaixadosDesceram{e}', inteiro(rb['n_desceram_na_lista']))
    with opcional(m, 'falhas do estrato B'):
        fb = d['falhas_estrato_b']
        m.add('EstFalhasN', inteiro(fb['nao_detectados']))
        m.add('EstFalhasTotal', inteiro(fb['total']))
        pf = fb['por_categoria_final']
        for chave, tok in (('sem_linguagem_toxica_nas_frases_amostradas', 'SemToxica'),
                           ('erro_do_classificador', 'ErroClassif'),
                           ('diluicao_na_agregacao', 'Diluicao'),
                           ('fora_do_escopo', 'ForaEscopo')):
            m.add(f'EstFalhas{tok}', inteiro(pf[chave]))
        m.add('EstFalhasPendentes', inteiro(fb['pendentes_autor']))
        # Categoria mecânica (do classificador, sem julgamento humano): é o que o texto
        # pode afirmar sem ler as frases. Diluição = alguma frase amostrada predita
        # Leve/Severa e a média acima do limiar; sem sinal = as três preditas Nenhuma.
        prov = fb['por_categoria_provisoria']
        m.add('EstFalhasDiluicaoProv', inteiro(prov.get('diluicao_na_agregacao', 0)),
              'perdas com alguma frase amostrada predita tóxica (média acima do limiar)')
        m.add('EstFalhasSemSinalProv',
              inteiro(prov.get('sem_linguagem_toxica_nas_frases_amostradas', 0)),
              'perdas com as três frases amostradas preditas Nenhuma')
        sev = [x for x in fb.get('detalhe', [])
               if x['categoria_provisoria'] == 'diluicao_na_agregacao'
               and any(f['classe_predita'] == 'Severa' for f in x['frases'])]
        m.add('EstFalhasDiluicaoComSevera', inteiro(len(sev)),
              'perdas por diluição em que uma frase amostrada foi predita Severa')
        if fb['pendentes_autor']:
            m.nota(f'{fb["pendentes_autor"]} falha(s) do estrato B sem categoria final por '
                   'julgamento humano; o texto usa só as categorias mecânicas (*Prov)')
    with opcional(m, 'rótulos humanos das frases amostradas'):
        rh = d['rotulos_humanos']
        if not rh:
            raise KeyError('rotulos_humanos ausente (rodar analisar_estratos com --rotulos-humanos)')
        c = rh['concordancia']
        m.add('EstHumFrases', inteiro(rh['n_frases']), 'anotação cega das frases do estudo de caso')
        m.add('EstHumKappa', dec(c['kappa_fleiss'], 3), 'Fleiss, três anotadores')
        m.add('EstHumUnanimes', inteiro(c['unanimes']))
        m.add('EstHumSemMaioria', inteiro(c['sem_maioria']))
        t = rh['total']
        m.add('EstHumAcuraciaPct', dec(100 * t['acuracia'], 1), 'humano x classe predita')
        m.add('EstHumMacroFUm', dec(t['macro_f1'], 2), 'três classes')
        m.add('EstHumBinFUm', dec(t['binario']['macro_f1'], 2), 'tóxica x não tóxica')
        for nome, e in rh['por_estrato'].items():
            X = nome[0]
            for k, tok in (('Nenhuma', 'Nenhuma'), ('Leve', 'Leve'), ('Severa', 'Severa')):
                m.add(f'EstHum{X}{tok}', inteiro(e['distribuicao_humano'][k]),
                      f'frases do estrato {X} com rótulo humano {k}')
            m.add(f'EstHum{X}Toxicas', inteiro(e['toxicas_humano']))
            m.add(f'EstHum{X}ToxSinalizadas', inteiro(e['toxicas_humano_sinalizadas']),
                  'tóxicas para os anotadores e preditas Leve/Severa')
            m.add(f'EstHum{X}NenhumaSinalizadas', inteiro(e['nenhuma_humano_sinalizadas']),
                  'Nenhuma para os anotadores e preditas Leve/Severa')
            m.add(f'EstHum{X}VideosTox', inteiro(e['videos_com_frase_toxica_humano']),
                  'vídeos com alguma frase amostrada tóxica para os anotadores')
            m.add(f'EstHum{X}AcuraciaPct', dec(100 * e['acuracia'], 1))
        ah = rh['auc_rotulo_humano_do_video']
        m.add('EstHumAuc', dec(ah['auc'], 3), 'score do vídeo x rótulo humano das mesmas frases')
        m.add('EstHumAucIcBaixo', dec(ah['auc_ic95'][0], 3))
        m.add('EstHumAucIcAlto', dec(ah['auc_ic95'][1], 3))
        m.add('EstHumAucPositivos', inteiro(ah['positivos']))
        m.add('EstHumAucNegativos', inteiro(ah['negativos']))
        fa = rh['falsos_alarme_a']
        m.add('EstHumFaErro', inteiro(fa['por_erro_do_classificador']),
              f'falsos alarmes de A no limiar designado cujas frases sinalizadas são Nenhuma '
              f'para os anotadores (de {fa["n"]})')
        # Limiares exploratórios em que nem todo vídeo de A cai abaixo (no de 0,95 todos
        # caem, e a linha nada diz sobre erros).
        n_a = rh['por_estrato']['A_responsavel']['videos']
        explor = [x for x in rh.get('falsos_alarme_a_por_limiar', [])
                  if abs(x['limiar'] - fa['limiar']) > 1e-9 and x['n'] < n_a]
        vids = {y['id']: y for x in explor for y in x['detalhe']}
        m.add('EstHumFaExplorVideos', inteiro(len(vids)),
              'vídeos de A em falso alarme em algum limiar exploratório (exceto onde todos caem)')
        m.add('EstHumFaExplorErro', inteiro(sum(y['erro_do_classificador'] for y in vids.values())),
              'desses, quantos só por frases que os anotadores rotularam Nenhuma')
    with opcional(m, 'sensibilidade a pesos e agregação'):
        ls = d['sensibilidade_pesos_agregacao']['linhas']
        aucs = [x['auc'] for x in ls]
        m.add('EstSensN', inteiro(len(ls)))
        m.add('EstSensAucMin', dec(min(aucs), 3))
        m.add('EstSensAucMax', dec(max(aucs), 3))
        da = [x['deslocamento_medio_abs_A'] for x in ls]
        m.add('EstSensDeslocAMin', dec(min(da), 1))
        m.add('EstSensDeslocAMax', dec(max(da), 1))


MODOS_LLM = {'livre:1': 'LivreUm', 'restrita:1': 'RestritaUm', 'restrita:2': 'RestritaDois',
             'restrita:4': 'RestritaQuatro', 'restrita:8': 'RestritaOito'}


def secao_latencia(m: Macros, res: Path, ctx: dict):
    d = carregar(m, res, None, 'latencia_*.json')
    m.secao('Latência por página de resultados (medir_latencia.py)')
    if d.get('_AVISO'):
        m.nota(f'ATENÇÃO: {d["_AVISO"]}')
    with opcional(m, 'configuração'):
        m.add('LatVideosPorPagina', inteiro(d['configuracao']['videos_por_pagina']))
    bert = d['bertimbau']
    # unidade da página: GPU em ms, CPU em s
    for disp, dt, unid, fator, casas in (('gpu', 'Gpu', 'Ms', 1000.0, 1), ('cpu', 'Cpu', 'S', 1.0, 2)):
        x = bert.get(disp) or {}
        if not x.get('paginas'):
            m.nota(f'BERTimbau {disp}: não medido ({x.get("pulado", "sem dados")})')
            continue
        with opcional(m, f'BERTimbau {disp}'):
            for k, kt in KS:
                p = x['paginas'][str(k)]
                m.add(f'LatBert{dt}{kt}{unid}', dec(p['mediana_s'] * fator, casas),
                      f'mediana por página ({p["n_frases"]} frases)')
                m.add(f'LatBert{dt}{kt}PNoventaCinco{unid}', dec(p['p95_s'] * fator, casas))
            m.add(f'LatBert{dt}PorFraseMs', dec(x['paginas']['12']['por_frase_mediana_ms'], 2),
                  'mediana da página de k = 12 / frases')
            if x.get('carga_modelo_s') is not None:
                m.add(f'LatBert{dt}CargaS', dec(x['carga_modelo_s'], 1), 'fora do tempo da página')
            if x.get('memoria_gpu_pico_mb') is not None:
                m.add(f'LatBert{dt}MemPicoMb', inteiro(x['memoria_gpu_pico_mb']))
    llm = d.get('llm') or {}
    if not llm.get('modos'):
        m.nota(f'LLM: não medido ({llm.get("pulado", "sem dados")})')
    else:
        m.add('LatLlmNome', tex_escape(llm.get('modelo', '')))
        frac = (llm.get('memoria') or {}).get('fracao_na_gpu')
        if frac is not None:
            m.add('LatLlmMemGpuPct', pct(frac, 0), 'fração do modelo na memória da GPU (/api/ps)')
        for modo, x in llm['modos'].items():
            mt = MODOS_LLM.get(modo)
            if mt is None:
                m.nota(f'modo do LLM sem macro definida: {modo}')
                continue
            with opcional(m, f'LLM {modo}'):
                for k, kt in KS:
                    p = x['paginas'][str(k)]
                    m.add(f'LatLlm{mt}{kt}S', dec(p['mediana_s'], 1),
                          f'mediana por página ({p["n_frases"]} frases)')
                    m.add(f'LatLlm{mt}{kt}PNoventaCincoS', dec(p['p95_s'], 1))
                    if p.get('n_erros'):
                        m.nota(f'LLM {modo} k={k}: {p["n_erros"]} requisições com erro')
                p3 = x['paginas']['3']
                if p3.get('requisicao_mediana_s') is not None:
                    m.add(f'LatLlm{mt}ReqMedianaS', dec(p3['requisicao_mediana_s'], 2))
                om = p3.get('ollama_mediana') or {}
                if om.get('eval_count') is not None:
                    m.add(f'LatLlm{mt}TokensKTres', inteiro(om['eval_count']),
                          'mediana de tokens gerados por resposta, k = 3')
                if om.get('total_ms') is not None:
                    # tempo do servidor por requisição: com c concorrentes, cresce com a
                    # fila quando o servidor atende uma requisição de cada vez
                    m.add(f'LatLlm{mt}ServidorKTresS', dec(om['total_ms'] / 1000, 2),
                          'mediana do total_duration do Ollama por requisição, k = 3')
                m.add(f'LatLlm{mt}PorFraseS', dec(x['paginas']['12']['por_frase_mediana_ms'] / 1000, 2))
                conc = (x.get('concordancia_com_livre_1') or {}).get('3')
                if conc is not None:
                    m.add(f'LatLlm{mt}ConcordKTres', pct(conc), 'classe igual à do modo livre:1')
        with opcional(m, 'razão LLM / BERTimbau'):
            lento = llm['modos']['livre:1']['paginas']['3']['mediana_s']
            rapido = bert['gpu']['paginas']['3']['mediana_s']
            m.add('LatRazaoLlmBertGpuKTres', inteiro(lento / rapido),
                  'mediana LLM (livre, sequencial) / mediana BERTimbau GPU, k = 3')
            m.add('LatRazaoOrdens', inteiro(math.floor(math.log10(lento / rapido))),
                  'ordens de grandeza (piso de log10 da razão)')
    with opcional(m, 'hardware'):
        hw = d['hardware']
        m.add('LatCpuNome', tex_escape(hw['cpu']))
        gpu = hw.get('gpu') or {}
        if gpu.get('nome'):
            m.add('LatGpuNome', tex_escape(gpu['nome']))
        if hw.get('memoria_ram_gb') is not None:
            m.add('LatRamGb', dec(hw['memoria_ram_gb'], 0))


def secao_scores(m: Macros, res: Path, ctx: dict):
    d = carregar(m, res, 'artigo', 'scores_exemplo.json')
    m.secao('Score de toxicidade verbal no teste (calcular_scores_exemplo.py)')
    dist = d['distribuicao_por_classe']
    n = sum(dist[c]['n'] for c in CLASSES)
    m.add('ScoreFrasesN', inteiro(n))
    registrar_n_teste(ctx, 'scores_exemplo', n)
    for cl in CLASSES:
        x = dist[cl]
        m.add(f'ScoreMediana{cl}', dec(x['mediana'], 3))
        m.add(f'ScoreQUm{cl}', dec(x['p25'], 3), 'primeiro quartil')
        m.add(f'ScoreQTres{cl}', dec(x['p75'], 3), 'terceiro quartil')
        m.add(f'ScoreMedia{cl}', dec(x['media'], 3))
    with opcional(m, 'vídeos da figura de demonstração'):
        s = [v['score_toxicidade'] for v in d['videos'].values()]
        m.add('AppVideosN', inteiro(len(s)))
        m.add('AppScoreMax', dec(max(s), 3))
        m.add('AppScoreMin', dec(min(s), 3))


def secao_teste_por_lote(m: Macros, res: Path, ctx: dict):
    d = carregar(m, res, None, 'teste_por_lote_*.json')
    m.secao('Teste por lote de anotação e época restaurada (analisar_teste_por_lote.py)')
    registrar_n_teste(ctx, 'teste_por_lote', d['global']['n_frases'])
    for lote, lt in (('1', 'Um'), ('2', 'Dois')):
        x = d['por_lote'][lote]
        m.add(f'LoteTeste{lt}N', inteiro(x['n_frases']))
        m.add(f'LoteTeste{lt}Videos', inteiro(x['n_videos']))
        m.add(f'LoteTeste{lt}SeveraPct', pct(x['distribuicao']['Severa'] / x['n_frases']),
              'frases Severa do lote no teste')
        b = x['bertimbau']
        m.add(f'BertLote{lt}FUm', dec(b['f1_macro_media'], 4), 'média das 5 sementes')
        m.add(f'BertLote{lt}Dp', dec(b['f1_macro_dp'], 4), 'DP entre sementes (ddof=1)')
        m.add(f'SvmAjLote{lt}FUm', dec(x['svm_ajustado']['f1_macro'], 4))
    m.add('LoteTesteVideosAmbos', inteiro(d['videos_com_os_dois_lotes_no_teste']),
          'vídeos do teste com frases dos dois lotes')
    with opcional(m, 'época restaurada'):
        e = d['epoca_restaurada']
        eps = list(e['por_semente'].values())
        m.add('EpocaMax', inteiro(e['epocas_maximas']), 'orçamento de épocas da configuração')
        m.add('EpocaRef', inteiro(e['por_semente']['42']), 'época restaurada, semente 42')
        m.add('EpocaMinSementes', inteiro(min(eps)))
        m.add('EpocaMaxSementes', inteiro(max(eps)))


VIZINHAS = ('3_palavras_min_3', '3_palavras_min_5', '2_palavras_min_4', '4_palavras_min_4')


def secao_quase_duplicatas(m: Macros, res: Path, ctx: dict):
    d = carregar(m, res, None, 'quase_duplicatas_*.json')
    m.secao('Grupos de quase-duplicatas (analisar_quase_duplicatas.py)')
    g = d['descricao_grupos']
    m.add('QdN', inteiro(g['n_frases']), g['regra'])
    m.add('QdTerminacoes', inteiro(g['n_terminacoes']))
    if 'terminacoes_rotulo_unico' in g:
        m.add('QdTerminacoesRotuloUnico', inteiro(g['terminacoes_rotulo_unico']),
              'terminações em que todas as frases têm o mesmo rótulo final')
        m.add('QdMaiorGrupo', inteiro(g['maior_grupo']), 'frases da terminação mais frequente')
    m.add('QdVideos', inteiro(g['n_videos']))
    m.add('QdNenhumaN', inteiro(g['distribuicao']['Nenhuma']))
    m.add('QdLeveN', inteiro(g['distribuicao']['Leve']))
    m.add('QdSeveraN', inteiro(g['distribuicao']['Severa']))
    m.add('QdLoteUmN', inteiro(g['por_lote'].get('1', 0)))
    m.add('QdLoteDoisN', inteiro(g['por_lote'].get('2', 0)))
    m.add('QdTreinoN', inteiro(g['por_particao'].get('treino', 0)), 'lado da partição por vídeo')
    m.add('QdTesteN', inteiro(g['por_particao'].get('teste', 0)))
    m.add('QdSemPalavraoN', inteiro(g['toxicas_sem_palavrao_no_grupo']))
    m.add('QdSemPalavraoPct', pct(g['toxicas_sem_palavrao_no_grupo'] / g['toxicas_sem_palavrao_no_corpus']),
          'das tóxicas sem palavrão do corpus publicado')
    m.add('ToxSemPalavraoPublicadoN', inteiro(g['toxicas_sem_palavrao_no_corpus']), 'corpus publicado')
    m.add('ToxicasPublicadoN', inteiro(g['toxicas_no_corpus']), 'corpus publicado')
    m.add('ToxSemPalavraoPublicadoPct', pct(g['toxicas_sem_palavrao_no_corpus'] / g['toxicas_no_corpus']))
    vz = g['regras_vizinhas']
    m.add('QdTresMinTresN', inteiro(vz['3_palavras_min_3']['n_frases']))
    m.add('QdTresMinTresNenhumaN', inteiro(vz['3_palavras_min_3']['n_nenhuma']))
    m.add('QdTresMinCincoN', inteiro(vz['3_palavras_min_5']['n_frases']))
    m.add('QdDoisMinQuatroN', inteiro(vz['2_palavras_min_4']['n_frases']))
    m.add('QdDoisMinQuatroNenhumaN', inteiro(vz['2_palavras_min_4']['n_nenhuma']))
    m.add('QdQuatroMinQuatroN', inteiro(vz['4_palavras_min_4']['n_frases']))
    te = d['teste']
    sens = [te[f'sem_{r}']['bertimbau_f1_media'] for r in VIZINHAS]
    extra = [te[f'sem_{r}']['frases_retiradas'] for r in VIZINHAS]
    m.add('QdSensMin', dec(min(sens), 4), 'F1 do teste sem as frases a mais das regras vizinhas')
    m.add('QdSensMax', dec(max(sens), 4))
    m.add('QdSensExtraMax', inteiro(max(extra)), 'frases do teste que a regra vizinha mais frouxa tiraria a mais')

    # Rodada video-v1, lida de propósito: o teste dela ainda continha as quase-duplicatas.
    pasta_v1 = experimento.NLP_ROOT / 'evaluation' / 'results' / 'video-v1'
    arq = mais_recente(str(pasta_v1 / 'quase_duplicatas_*.json'))
    if arq is None:
        m.ausente('Quase-duplicatas na rodada video-v1', f'nenhum quase_duplicatas_*.json em {_rel(pasta_v1)}')
        return
    m.fonte = arq
    v1 = ler_json(arq)
    m.secao('Quase-duplicatas na rodada video-v1 (com elas no treino e no teste)')
    m.nota('Lido da rodada video-v1 de propósito: mede o quanto as quase-duplicatas inflavam o teste.')
    t1 = v1['teste']
    m.add('VumBertFUm', dec(t1['inteiro']['bertimbau_f1_media'], 4), 'teste inteiro (417 frases)')
    m.add('VumBertSemQdFUm', dec(t1['sem_3_palavras_min_4']['bertimbau_f1_media'], 4),
          'mesmo modelo, teste sem as quase-duplicatas')
    m.add('VumSvmFUm', dec(t1['inteiro']['svm_ajustado_f1'], 4))
    m.add('VumSvmSemQdFUm', dec(t1['sem_3_palavras_min_4']['svm_ajustado_f1'], 4))
    m.add('VumTesteN', inteiro(t1['inteiro']['n_frases']))
    m.add('VumTesteSemQdN', inteiro(t1['sem_3_palavras_min_4']['n_frases']))
    a = v1['acuracia_quase_duplicatas_teste']
    b = v1['acuracia_demais_toxicas_teste']
    m.add('VumAcertoQdPct', pct(a['bertimbau_media']), f"{a['n_frases']} quase-duplicatas do teste")
    m.add('VumAcertoDemaisToxPct', pct(b['bertimbau_media']), f"{b['n_frases']} demais tóxicas do teste")
    m.add('VumAcertoQdSvmPct', pct(a['svm_ajustado']))
    m.add('VumAcertoDemaisToxSvmPct', pct(b['svm_ajustado']))
    todas = [t1[k]['bertimbau_f1_media'] for k in t1 if k.startswith('sem_')]
    m.add('VumSemQdMin', dec(min(todas), 4), 'F1 do teste v1 sem as frases de cada regra (principal e vizinhas)')
    m.add('VumSemQdMax', dec(max(todas), 4))


def secao_consistencia(m: Macros, ctx: dict, com_resultados: bool):
    m.secao('Conferência entre as fontes (sem macros)')
    ns = ctx.get('n_teste', {})
    if ns:
        distintos = sorted(set(ns.values()))
        if len(distintos) > 1:
            m.nota('ATENÇÃO: n do teste difere entre as fontes: '
                   + ', '.join(f'{k}={v}' for k, v in ns.items()))
        else:
            m.nota(f'n do teste igual em todas as fontes ({distintos[0]}): {", ".join(ns)}')
    tags = sorted(set(m.tags.values()))
    if len(tags) > 1:
        m.nota('ATENÇÃO: JSON de rodadas diferentes: '
               + '; '.join(f'{t}: ' + ', '.join(os.path.basename(k) for k, v in m.tags.items()
                                                 if v == t) for t in tags))
    elif tags and not com_resultados and tags[0] != experimento.tag():
        m.nota(f'ATENÇÃO: os JSON são da rodada {tags[0]}, não da atual ({experimento.tag()})')
    elif tags:
        m.nota(f'rodada gravada nos JSON: {tags[0]}')


SECOES_RESULTADO = [
    ('Busca de hiperparâmetros', secao_busca),
    ('Modelo final', secao_modelo_final),
    ('Avaliação no teste', secao_avaliacao),
    ('Baselines fixas', secao_baselines),
    ('Baselines ajustadas', secao_baselines_ajustadas),
    ('Dispersão', secao_dispersao),
    ('Comparação pareada', secao_comparacao),
    ('Teste por lote', secao_teste_por_lote),
    ('Quase-duplicatas', secao_quase_duplicatas),
    ('Referências sem ajuste fino', secao_sem_finetuning),
    ('LLM generativo', secao_llm),
    ('Detoxify', secao_detoxify),
    ('Limiar OOF e calibração', secao_calibracao),
    ('Pontos de operação (exploratório)', secao_pontos_operacao),
    ('Simulação de amostragem', secao_simulacao),
    ('Estudo de caso: estratos', secao_estratos),
    ('Latência', secao_latencia),
    ('Score de toxicidade verbal', secao_scores),
]


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Gera numeros.tex com os números citados no artigo')
    ap.add_argument('--resultados', default=None,
                    help='pasta dos JSON dos modelos (padrão: experimento.dir_resultados()); '
                         'corpus, anotação e partição continuam lidos da rodada atual')
    ap.add_argument('--saida', default=None,
                    help='arquivo .tex de saída (padrão: experimento.dir_latex("")/numeros.tex)')
    args = ap.parse_args()

    res_dados = experimento.dir_resultados()
    res = Path(args.resultados).resolve() if args.resultados else res_dados
    if not res.is_dir():
        raise SystemExit(f'pasta de resultados não encontrada: {res}')
    m = Macros()
    if args.resultados:
        m.origem = res
    rodar_secao(m, 'Corpus', secao_corpus, res_dados)
    rodar_secao(m, 'Anotação', secao_anotacao, res_dados)
    rodar_secao(m, 'Partição por vídeo', secao_particao)
    ctx = {}
    for titulo, fn in SECOES_RESULTADO:
        rodar_secao(m, titulo, fn, res, ctx)
    secao_consistencia(m, ctx, bool(args.resultados))

    destino = Path(args.saida) if args.saida else experimento.dir_latex('') / 'numeros.tex'
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(m.tex(), encoding='utf-8')
    total = sum(len(s['itens']) for s in m.secoes)
    ausentes = [s['titulo'] for s in m.secoes if s['motivo']]
    com_nota = [s['titulo'] for s in m.secoes if s['notas'] and not s['motivo']]
    print(f'{total} macros em {destino}')
    if ausentes:
        print('Seções ausentes: ' + '; '.join(ausentes))
    if com_nota:
        print('Seções com NOTA (veja os comentários no .tex): ' + '; '.join(com_nota))


if __name__ == '__main__':
    main()
