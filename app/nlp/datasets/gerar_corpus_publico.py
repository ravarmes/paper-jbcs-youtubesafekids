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
  tem_palavrao  -> os 17,1% com palavrao e a regra ingenua do lexico

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

from app.nlp.evaluation.scripts.analisar_corpus import PALAVROES  # noqa: E402

COLUNAS_SAIDA = [
    'id_sentenca', 'video_id', 'titulo', 'parte', 'link', 'frase',
    'n_palavras', 'tem_palavrao', 'tox', 'p1', 'p2', 'a1',
]


def gerar(entrada, saida):
    with open(entrada, encoding='utf-8-sig', newline='') as f:
        linhas = list(csv.DictReader(f, delimiter=';'))

    if not linhas:
        raise SystemExit(f'corpus vazio ou ilegivel: {entrada}')
    if 'FRASE' not in linhas[0]:
        raise SystemExit(f'{entrada} nao tem a coluna FRASE -- '
                         'e provavel que ja seja a versao publica')

    os.makedirs(os.path.dirname(os.path.abspath(saida)) or '.', exist_ok=True)
    with open(saida, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS_SAIDA, delimiter=';')
        w.writeheader()
        for i, r in enumerate(linhas, start=1):
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
            })

    print(f'{len(linhas)} sentencas -> {saida}')
    print('colunas publicadas:', ', '.join(COLUNAS_SAIDA))


def main():
    p = argparse.ArgumentParser(description='Gera a versao publica do corpus')
    p.add_argument('--entrada', required=True,
                   help='CSV completo, com a coluna FRASE')
    p.add_argument('--saida', default='data/corpus_toxicidade_publico.csv')
    a = p.parse_args()
    gerar(a.entrada, a.saida)


if __name__ == '__main__':
    main()
