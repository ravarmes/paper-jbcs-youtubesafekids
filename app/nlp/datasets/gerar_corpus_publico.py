"""
Gera a versao publica do corpus de toxicidade.

O QUE PUBLICA
-------------
Tudo: a frase transcrita, o identificador e o titulo do video, a posicao de onde a
sentenca saiu, as tres anotacoes individuais e o rotulo final. Publicar o texto e o
que torna o recurso reutilizavel -- sem ele o corpus permite conferir as estatisticas
do artigo, mas nao retreinar nem reavaliar nada, e um corpus anotado que nao pode ser
usado para treinar nao e um recurso, e um relatorio.

Duas colunas derivadas continuam saindo daqui, agora por conveniencia e nao por
necessidade, com as mesmas definicoes de analisar_corpus.py:

  n_palavras    -> a media e a mediana de palavras da Tabela 1
  tem_palavrao  -> as frases com palavrao e a regra ingenua do lexico (Tabela 1)

E uma marca:

  quase_duplicata -> 1 se a frase pertence a um grupo de quase-duplicatas (as 3
                     ultimas palavras se repetem em 4 ou mais frases do corpus; regra em
                     app/nlp/datasets/quase_duplicatas.py). Essas frases ficam fora dos
                     experimentos do artigo (rodada v2), mas continuam publicadas.

O QUE NAO PUBLICA
-----------------
Nada e removido do corpus. As frases sao trechos curtos de transcricoes de videos de
terceiros, citados aqui para fim de pesquisa; nenhum video e redistribuido, e o link
de origem acompanha cada linha para que a proveniencia seja verificavel.

Uso:
    python -m app.nlp.datasets.gerar_corpus_publico         --entrada ../dados-completos/corpus_toxicidade_completo.csv         --saida data/corpus_toxicidade_publico.csv
"""

import argparse
import csv
import os
import re
import sys

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
sys.path.insert(0, project_root)

# O léxico vem de analisar_corpus.py, lido pelo caminho do arquivo: importar o pacote
# app.nlp.evaluation carrega o torch (via __init__), e este script não precisa dele.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'analisar_corpus',
    os.path.join(project_root, 'app', 'nlp', 'evaluation', 'scripts', 'analisar_corpus.py'))
_analisar_corpus = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_analisar_corpus)
PALAVROES = _analisar_corpus.PALAVROES

from app.nlp.datasets import quase_duplicatas  # noqa: E402

COLUNAS_SAIDA = [
    'id_sentenca', 'video_id', 'titulo', 'parte', 'link', 'frase',
    'n_palavras', 'tem_palavrao', 'tox', 'p1', 'p2', 'a1', 'lote', 'quase_duplicata',
]

# Lote de anotação de cada frase (1 = primeiro lote, 2 = segundo), lido do mapa de
# origem gravado por documentos/anotacao-substituicao/montar_corpus_v1.py: origem
# 'v0:<id>' é do primeiro lote, 'substituicao:F<item>' do segundo. O mapa fica fora de src/
# (tem os identificadores internos das planilhas); só o número do lote é publicado.
ORIGEM_PADRAO = os.path.join(project_root, '..', 'dados-completos', 'corpus-v1',
                             'origem_frases.csv')


def ler_lotes(caminho):
    with open(caminho, encoding='utf-8-sig', newline='') as f:
        linhas = list(csv.DictReader(f, delimiter=';'))
    lotes = {}
    for r in linhas:
        origem = r['origem']
        if origem.startswith('v0:'):
            lotes[int(r['id_sentenca'])] = (1, r['video_id'])
        elif origem.startswith('substituicao:F'):
            lotes[int(r['id_sentenca'])] = (2, r['video_id'])
        else:
            raise SystemExit(f'origem desconhecida em {caminho}: {origem!r}')
    return lotes


def gerar(entrada, saida, origem=ORIGEM_PADRAO):
    with open(entrada, encoding='utf-8-sig', newline='') as f:
        linhas = list(csv.DictReader(f, delimiter=';'))
    lotes = ler_lotes(origem)
    if len(lotes) != len(linhas):
        raise SystemExit(f'mapa de origem com {len(lotes)} linhas, corpus com {len(linhas)}')

    if not linhas:
        raise SystemExit(f'corpus vazio ou ilegivel: {entrada}')
    if 'FRASE' not in linhas[0]:
        raise SystemExit(f'{entrada} nao tem a coluna FRASE -- '
                         'e provavel que ja seja a versao publica')

    marca = quase_duplicatas.marcar([(r.get('FRASE') or '').strip('" ').strip() for r in linhas])

    os.makedirs(os.path.dirname(os.path.abspath(saida)) or '.', exist_ok=True)
    with open(saida, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS_SAIDA, delimiter=';')
        w.writeheader()
        for i, r in enumerate(linhas, start=1):
            if lotes[i][1] != r.get('ID', ''):
                raise SystemExit(f'linha {i}: o mapa de origem diz video {lotes[i][1]}, '
                                 f'o corpus diz {r.get("ID")} -- mapa desatualizado?')
            frase = (r.get('FRASE') or '').strip('" ').strip()
            w.writerow({
                'id_sentenca': i,
                'video_id': r.get('ID', ''),
                'titulo': r.get('TITULO', ''),
                'parte': r.get('PARTE', ''),
                'link': r.get('LINK', ''),
                'frase': frase,
                'n_palavras': len(frase.split()),
                'tem_palavrao': int(bool(re.search(PALAVROES, frase.lower()))),
                'tox': r.get('TOX', ''),
                'p1': r.get('P1', ''),
                'p2': r.get('P2', ''),
                'a1': r.get('A1', ''),
                'lote': lotes[i][0],
                'quase_duplicata': int(marca[i - 1]),
            })

    print(f'{len(linhas)} sentencas -> {saida}')
    print('colunas publicadas:', ', '.join(COLUNAS_SAIDA))


def main():
    p = argparse.ArgumentParser(description='Gera a versao publica do corpus')
    p.add_argument('--entrada', required=True,
                   help='CSV completo, com a coluna FRASE')
    p.add_argument('--saida', default='data/corpus_toxicidade_publico.csv')
    p.add_argument('--origem', default=ORIGEM_PADRAO,
                   help='mapa id_sentenca -> origem (lote de anotacao)')
    a = p.parse_args()
    gerar(a.entrada, a.saida, a.origem)


if __name__ == '__main__':
    main()
