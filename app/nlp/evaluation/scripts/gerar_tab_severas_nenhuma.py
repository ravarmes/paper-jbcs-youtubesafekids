"""
Tabela das frases Severas do teste que o modelo de referência classificou como
Nenhuma (comentário menor 6 do revisor H: "mostre as frases").

Lê o JSON mais recente de analisar_calibracao_limiar.py (campo
`severas_preditas_nenhuma`), que já traz o texto de cada frase, as três
probabilidades, o score e se o limiar escolhido no OOF a sinaliza. Nenhum
número é digitado aqui.

O que é escrito à mão neste arquivo são só as TRADUÇÕES de leitura (convenção
da Seção 1.3 do artigo: frase original em itálico, tradução dos autores entre
aspas simples, que não entra em nenhuma etapa do pipeline). Elas são casadas
pelo texto exato da frase; frase sem tradução sai com a célula vazia e um aviso.

Saída: tabelas/tab-severas-nenhuma.tex em experimento.dir_latex(...).

Uso (de src/; pelo caminho do arquivo, para não importar o torch que
app/nlp/evaluation/__init__.py carrega):
    python app/nlp/evaluation/scripts/gerar_tab_severas_nenhuma.py
"""

import glob
import json
import os
import sys

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from app.nlp import experimento  # noqa: E402

# Traduções de leitura (autores). Chave: texto exato da frase no corpus.
TRADUCOES = {
    'voce e a maluca': "you are the crazy one",
    'sortudo desgraçado.': "lucky bastard",
    'tu ta ferrado': "you're screwed",
    'chupa meu saco': "suck my balls",
    'Essa é a única forma de manter esses criollos sob controle.':
        "this is the only way to keep these \\textit{criollos} under control",
    'diabo Eu nunca disse Mas você tatua na sua bunda':
        "damn, I never said it, but you get it tattooed on your butt",
    'Usar sua bunda como chapeu': "wear your butt as a hat",
    'Me chupa': "blow me",
    'voce que me forço a isso': "you are the one who forced me into this",
    'Porque não, a Cusão': "because no, you asshole",
    'Não quero voces aqui': "I don't want you (all) here",
    'eu cozinho voce': "I'll cook you",
    'pais otarios': "sucker parents",
    'Piranhazinha': "little slut",
    'Boa noite, o amigo imaginário do Henry passou a mão na minha bunda.':
        "good night, Henry's imaginary friend groped my butt",
}


def _mais_recente(padrao):
    arquivos = sorted(glob.glob(padrao), key=os.path.getmtime)
    return arquivos[-1] if arquivos else None


def _tex(texto):
    trocas = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
              '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
              '^': r'\textasciicircum{}'}
    return ''.join(trocas.get(c, c) for c in texto)


def _dec(x, casas):
    return f'{x:.{casas}f}'


def main():
    res = experimento.dir_resultados()
    arq = _mais_recente(os.path.join(str(res), 'calibracao_limiar_*.json'))
    if arq is None:
        sys.exit('calibracao_limiar_*.json não encontrado; rode analisar_calibracao_limiar.py')
    with open(arq, encoding='utf-8') as f:
        d = json.load(f)
    bloco = d['severas_preditas_nenhuma']
    if not bloco.get('textos_incluidos'):
        sys.exit(f'O JSON não traz os textos ({bloco.get("motivo_textos_omitidos")}).')
    frases = sorted(bloco['frases'], key=lambda r: r['score'])
    tau = bloco['tau_oof_usado']
    n_sinal = sum(1 for r in frases if r['sinalizada_pelo_tau_oof'])

    linhas = [
        '% NAO EDITAR A MAO: gerado por gerar_tab_severas_nenhuma.py a partir de',
        f'% {os.path.basename(arq)} (campo severas_preditas_nenhuma).',
        '\\begin{table*}[!t]',
        '\\centering',
        '\\small',
        f'\\caption{{The {bloco["n"]} test sentences annotated Severe that the reference model '
        f'classifies as None (of {bloco["n_severa_teste"]} Severe test sentences), ordered by '
        f'verbal-toxicity score. Sentences are reproduced verbatim, as transcribed; the '
        f'translations are a reading aid by the authors. $P$: class probabilities of the '
        f'model; flagged: whether $P(\\text{{Severe}}) > \\tau = {tau:.2f}$, the threshold '
        f'chosen on out-of-fold predictions (Section~\\ref{{sec:operacao}}), which flags '
        f'{n_sinal} of them.}}',
        '\\label{tab:severas-nenhuma}',
        '\\begin{tabular}{@{}p{0.32\\textwidth}p{0.31\\textwidth}cccc@{}}',
        '\\toprule',
        'Sentence (original) & Translation & $P(\\text{None})$ & $P(\\text{Severe})$ & '
        'Score & Flagged \\\\',
        '\\midrule',
    ]
    faltando = []
    for r in frases:
        trad = TRADUCOES.get(r['texto'])
        if trad is None:
            faltando.append(r['texto'])
            trad = ''
        linhas.append(
            f'\\textit{{{_tex(r["texto"])}}} & `{trad}\' & {_dec(r["p_nenhuma"], 2)} & '
            f'{_dec(r["p_severa"], 2)} & {_dec(r["score"], 3)} & '
            f'{"yes" if r["sinalizada_pelo_tau_oof"] else "no"} \\\\')
    linhas += ['\\bottomrule', '\\end{tabular}', '\\end{table*}', '']

    destino = experimento.dir_latex('tabelas') / 'tab-severas-nenhuma.tex'
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text('\n'.join(linhas), encoding='utf-8')
    print(f'tab-severas-nenhuma.tex -> {destino}')
    for t in faltando:
        print(f'AVISO: frase sem tradução de leitura: {t!r}')


if __name__ == '__main__':
    main()
