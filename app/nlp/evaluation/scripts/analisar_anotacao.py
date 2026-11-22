"""
Concordância entre os anotadores e composição do corpus por vídeo.

Responde a quatro pedidos da revisão do JBCS, todos sem modelo treinado:
  - confusão por par de anotadores e kappa por classe (M2h);
  - frases com três rótulos distintos e como foram resolvidas (M2i);
  - frases por vídeo e classes por vídeo (M1c);
  - vídeos e frases por franquia, e frases por parte do vídeo (m1).

Saídas:
  results/<tag>/anotacao/anotacao_stats.json
  <latex>/tabelas/tab-concordancia.tex  (confusão por par + kappa por classe)
  <latex>/tabelas/tab-videos.tex        (frases por vídeo, classes por vídeo, franquias)
Com YSK_ESCREVER_LATEX=1, as tabelas vão para latex/tabelas; sem isso, para
results/<tag>/latex/tabelas (ver experimento.dir_latex).

Uso (de dentro de src/):
    python -m app.nlp.evaluation.scripts.analisar_anotacao
"""

import json
import os
import re
import sys
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp import experimento  # noqa: E402
from app.nlp.datasets.prepare_data_toxicity import load_toxicity_corpus  # noqa: E402

CLASSES = ['Nenhuma', 'Leve', 'Severa']
NOMES_EN = ['None', 'Mild', 'Severe']
ANOTADORES = ['P1', 'P2', 'A1']
# Papéis dos anotadores (autores, 2026-11-19): P1 professor, P2 psicóloga, A1 aluno.
PAPEL_EN = {'P1': 'professor', 'P2': 'psychologist', 'A1': 'student'}

# Franquias identificadas pelo título do vídeo (expressões regulares, sem acento,
# minúsculas). Um vídeo que não casa com nenhuma entra em "outros".
FRANQUIAS = [
    ('Rick and Morty', r'rick (e|and|&) morty|rick and morty|rick.*morty|vingar de rick'),
    ('South Park', r'south ?park|cartman'),
    ('The Simpsons', r'simpson'),
    ('Family Guy', r'family guy|uma familia da pesada|peter griffin'),
    ('Brickleberry', r'brickleberry'),
    ('The Boondocks', r'boondocks'),
    ('Aqua Teen Hunger Force', r'aqua ?teen|mestre shake'),
    ('BoJack Horseman', r'bojack'),
    ('American Dad', r'american dad'),
    ('Big Mouth', r'big mouth'),
    ('Futurama', r'futurama'),
    ('Peppa Pig', r'peppa'),
    ('Show da Luna', r'show da luna'),
    ('SpongeBob', r'bob esponja|spongebob'),
    ('Looney Tunes', r'looney tunes|pernalonga|patolino'),
    ('Os Amiguinhos (fables)', r'os amiguinhos'),
    ('Galinha Pintadinha', r'galinha pintadinha'),
]
ANIMACAO_ADULTA = {'Rick and Morty', 'South Park', 'The Simpsons', 'Family Guy',
                   'Brickleberry', 'The Boondocks', 'Aqua Teen Hunger Force',
                   'BoJack Horseman', 'American Dad', 'Big Mouth', 'Futurama'}


def sem_acento(s):
    import unicodedata
    s = unicodedata.normalize('NFKD', str(s).lower())
    return ''.join(c for c in s if not unicodedata.combining(c))


def franquia(titulo):
    t = sem_acento(titulo)
    for nome, padrao in FRANQUIAS:
        if re.search(padrao, t):
            return nome
    return 'outros'


def cohen_kappa(a, b, k=3):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in range(k))
    return float((po - pe) / (1 - pe))


def fleiss_por_classe(anot):
    """Kappa de Fleiss de cada categoria (Fleiss, 1971, eq. 10)."""
    n_itens, n_anot = anot.shape
    contagem = np.stack([(anot == c).sum(axis=1) for c in range(3)], axis=1)
    p_j = contagem.sum(axis=0) / (n_itens * n_anot)
    saida = {}
    for c in range(3):
        num = (contagem[:, c] * (n_anot - contagem[:, c])).sum()
        den = n_itens * n_anot * (n_anot - 1) * p_j[c] * (1 - p_j[c])
        saida[CLASSES[c]] = float(1 - num / den)
    return saida


def tipo_desacordo(linha):
    """Par de classes envolvido num desacordo de duas contra uma."""
    usados = sorted(set(linha))
    if len(usados) == 3:
        return 'tres_classes'
    return '-'.join(CLASSES[c] for c in usados)


def main():
    df = load_toxicity_corpus()
    anot = df[ANOTADORES].astype(int).values
    stats = {'n_frases': int(len(df)), 'n_videos': int(df['ID'].nunique()),
             'papeis': {a: PAPEL_EN[a] for a in ANOTADORES}}

    # --- M2h: confusão e kappa por par; kappa por classe; onde está o desacordo
    pares = {}
    for a, b in combinations(ANOTADORES, 2):
        m = pd.crosstab(df[a], df[b]).reindex(index=range(3), columns=range(3), fill_value=0)
        pares[f'{a}-{b}'] = {
            'confusao': m.values.tolist(),
            'acordo': float((df[a] == df[b]).mean()),
            'cohen_kappa': cohen_kappa(df[a], df[b]),
        }
    stats['pares'] = pares
    stats['fleiss_por_classe'] = fleiss_por_classe(anot)
    desacordo = [tipo_desacordo(l) for l in anot if len(set(l)) > 1]
    stats['desacordos'] = {'n': len(desacordo), 'por_par_de_classes': dict(Counter(desacordo))}

    # --- M2i: frases com três rótulos distintos
    tres = df[[len(set(l)) == 3 for l in anot]]
    stats['tres_rotulos'] = [
        {'id_sentenca': int(r.id_sentenca), 'texto': r.texto,
         'anotacoes': {a: CLASSES[int(getattr(r, a))] for a in ANOTADORES},
         'rotulo_final': CLASSES[int(r.label)]}
        for r in tres.itertuples()]
    stats['regra_rotulo_final'] = 'moda das três anotações; sem moda, vale o P1 (professor)'

    # --- M1c: frases e classes por vídeo
    por_video = df.groupby('ID')
    n_por_video = por_video.size()
    classes_por_video = por_video['label'].agg(lambda s: tuple(sorted(set(s))))
    combos = Counter('+'.join(CLASSES[c] for c in t) for t in classes_por_video)
    stats['frases_por_video'] = {
        'distribuicao': {int(k): int(v) for k, v in n_por_video.value_counts().sort_index().items()},
        'media': float(n_por_video.mean()), 'mediana': float(n_por_video.median()),
        'min': int(n_por_video.min()), 'max': int(n_por_video.max()),
    }
    stats['classes_por_video'] = dict(combos.most_common())
    stats['videos_com_as_tres_classes'] = int(sum(len(t) == 3 for t in classes_por_video))

    # --- m1: franquias e partes do vídeo
    corpus = pd.read_csv(experimento.DATA_DIR / 'corpus_toxicidade_publico.csv', sep=';',
                         encoding='utf-8-sig', dtype=str)
    if len(corpus) != len(df) or not (corpus['video_id'].values == df['ID'].astype(str).values).all():
        raise SystemExit('corpus público desalinhado com o corpus de trabalho')
    vid = corpus.drop_duplicates('video_id').set_index('video_id')['titulo'].map(franquia)
    corpus['franquia'] = corpus['video_id'].map(vid)
    fr = (corpus.groupby('franquia')
          .agg(videos=('video_id', 'nunique'), frases=('frase', 'size'))
          .sort_values(['videos', 'frases'], ascending=False))
    stats['franquias'] = {k: {'videos': int(v.videos), 'frases': int(v.frases)}
                          for k, v in fr.iterrows()}
    adulta = corpus['franquia'].isin(ANIMACAO_ADULTA)
    stats['animacao_adulta'] = {
        'videos': int(corpus.loc[adulta, 'video_id'].nunique()),
        'frases': int(adulta.sum()),
        'prevalencia_toxica': float((corpus.loc[adulta, 'tox'] != 'Nenhuma').mean()),
        'prevalencia_toxica_demais': float((corpus.loc[~adulta, 'tox'] != 'Nenhuma').mean()),
    }
    stats['partes'] = corpus['parte'].value_counts().to_dict()

    # --- Lotes de anotação: mesmos três anotadores e mesmo guia, em dois lotes.
    # Concordância e composição de cada lote, para que a diferença entre eles fique
    # visível em vez de diluída no kappa do corpus inteiro.
    if 'lote' in corpus.columns:
        # Pelo caminho do arquivo: importar o pacote app.nlp.evaluation carrega o torch
        # (via __init__), e este script não precisa dele.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'analisar_corpus', os.path.join(os.path.dirname(__file__), 'analisar_corpus.py'))
        _ac = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_ac)
        fleiss_kappa = _ac.fleiss_kappa
        lotes = {}
        videos_lote1 = set(corpus.loc[corpus['lote'] == '1', 'video_id'])
        for lote, sub in corpus.groupby('lote'):
            idx = sub.index.values
            a = anot[idx]
            contagem = np.stack([(a == c).sum(axis=1) for c in range(3)], axis=1)
            lotes[str(lote)] = {
                'frases': int(len(sub)),
                'videos': int(sub['video_id'].nunique()),
                'videos_fora_do_lote1': int(len(set(sub['video_id']) - videos_lote1)),
                'distribuicao': {c: int((sub['tox'] == c).sum()) for c in CLASSES},
                'prevalencia_toxica': float((sub['tox'] != 'Nenhuma').mean()),
                'fleiss_kappa': float(fleiss_kappa(contagem)),
                'unanimidade': float(np.mean([len(set(l)) == 1 for l in a])),
            }
        stats['lotes'] = lotes

    pasta = experimento.dir_resultados('anotacao')
    (pasta / 'anotacao_stats.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                                               encoding='utf-8')
    escrever_tabelas(stats)
    print(json.dumps({k: v for k, v in stats.items() if k != 'tres_rotulos'},
                     indent=2, ensure_ascii=False))
    print(f'\nSalvo em {pasta}')


def pct(x):
    return f'{100 * x:.1f}\\%'


def escrever_tabelas(stats):
    tabelas = experimento.dir_latex('tabelas')
    cab = ('% Gerado por app/nlp/evaluation/scripts/analisar_anotacao.py\n'
           '% NAO EDITAR A MAO: reexecute o script apos mudar o corpus.\n')

    # Concordância: três matrizes lado a lado + kappa de Cohen por par + kappa por classe
    linhas = []
    for par, d in stats['pares'].items():
        a, b = par.split('-')
        # Papéis dos anotadores ficam na legenda: a linha precisa caber numa coluna.
        titulo = (f'{a} $\\times$ {b}: agreement {pct(d["acordo"])}, '
                  f'Cohen\'s $\\kappa$ = {d["cohen_kappa"]:.4f}')
        linhas.append(f'\\multicolumn{{4}}{{l}}{{{titulo}}} \\\\')
        linhas.append('\\midrule')
        linhas.append(f'{a} $\\downarrow$ / {b} $\\rightarrow$ & ' + ' & '.join(NOMES_EN) + ' \\\\')
        for i, nome in enumerate(NOMES_EN):
            linhas.append(f'{nome} & ' + ' & '.join(f'{v:,}' for v in d['confusao'][i]) + ' \\\\')
        linhas.append('\\midrule')
    kc = stats['fleiss_por_classe']
    linhas.append('Fleiss\' $\\kappa$ by class & '
                  + ' & '.join(f'{kc[c]:.3f}' for c in CLASSES) + ' \\\\')
    corpo = '\n'.join(linhas)
    papeis = ', '.join(f'{k} {v}' for k, v in PAPEL_EN.items())
    (tabelas / 'tab-concordancia.tex').write_text(
        cab + '\\begin{table}[!t]\n\\centering\n\\small\n'
        '\\caption{Confusion between each pair of raters over the full corpus, Cohen\'s '
        f'kappa per pair and category-wise Fleiss\' kappa ({papeis}).}}\n'
        '\\label{tab:concordancia}\n'
        '\\begin{tabular}{lrrr}\n\\toprule\n' + corpo + '\n\\bottomrule\n\\end{tabular}\n'
        '\\end{table}\n', encoding='utf-8')

    # Vídeos: frases por vídeo, classes por vídeo, franquias
    fpv = stats['frases_por_video']
    cpv = stats['classes_por_video']
    trad = dict(zip(CLASSES, NOMES_EN))

    def _n(qtd, sing, plur):
        return f'{qtd:,} {sing if qtd == 1 else plur}'

    # Frases por vídeo em faixas: a distribuição completa fica no JSON.
    faixas = [(1, 5), (6, 6), (7, 12), (13, 19), (20, 10**6)]
    linhas = []
    for lo, hi in faixas:
        n = sum(v for k, v in fpv['distribuicao'].items() if lo <= k <= hi)
        if not n:
            continue
        rot = (f'{lo}' if lo == hi else
               f'{lo} or more' if hi >= 10**6 else
               f'{lo}--{hi}')
        linhas.append(f'Videos with {rot} sentences & {n} \\\\')
    linhas += [f'Mean / median / max.\\ sentences per video & '
               f'{fpv["media"]:.1f} / {fpv["mediana"]:.0f} / {fpv["max"]} \\\\',
               '\\midrule']
    for combo, n in cpv.items():
        nome = ' + '.join(trad[c] for c in combo.split('+'))
        rot = f'only {nome}' if '+' not in combo else nome
        linhas.append(f'Videos with {rot} & {n} \\\\')
    linhas.append('\\midrule')
    for nome, d in stats['franquias'].items():
        if nome == 'outros':
            continue
        marca = '$^\\dagger$' if nome in ANIMACAO_ADULTA else ''
        linhas.append(f'{nome}{marca} & {_n(d["videos"], "video", "videos")}, '
                      f'{_n(d["frases"], "sentence", "sentences")} \\\\')
    outros = stats['franquias'].get('outros')
    if outros:
        linhas.append(f'Other titles & {_n(outros["videos"], "video", "videos")}, '
                      f'{_n(outros["frases"], "sentence", "sentences")} \\\\')
    adulta = stats['animacao_adulta']
    linhas += ['\\midrule',
               f'All adult animation$^\\dagger$ & {_n(adulta["videos"], "video", "videos")}, '
               f'{_n(adulta["frases"], "sentence", "sentences")} \\\\']
    corpo = '\n'.join(linhas)
    (tabelas / 'tab-videos.tex').write_text(
        cab + '\\begin{table}[!t]\n\\centering\n\\small\n'
        '\\caption{Composition of the corpus by video: sentences per video, classes present '
        'in each video, and videos per franchise as identified from the video title '
        '($^\\dagger$adult animation).}\n'
        '\\label{tab:videos}\n\\begin{tabular}{p{0.54\\columnwidth}p{0.34\\columnwidth}}\n'
        '\\toprule\n' + corpo + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n',
        encoding='utf-8')


if __name__ == '__main__':
    main()
