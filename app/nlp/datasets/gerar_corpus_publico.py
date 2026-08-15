"""
Gera a versao publica do corpus: tudo menos o texto das sentencas.

Por que existe
--------------
Os autores optaram por nao redistribuir as falas transcritas dos videos. O
corpus publicado traz, portanto, o identificador do video, a posicao de onde a
sentenca saiu, os tres rotulos individuais e o rotulo final -- mas nao a frase.

O custo dessa escolha e que o corpus publicado nao permite retreinar o modelo:
a anotacao e sobre a transcricao, e a transcricao nao e recuperavel a partir do
id do video (transcrever de novo, com outra ferramenta, produz outras strings).
Para atenuar isso, o arquivo publico carrega duas colunas derivadas que
permitem **verificar** as afirmacoes do artigo sobre o corpus sem reconstruir
o texto:

  n_palavras    -> reproduz a media de 8,8 palavras e a mediana 7
  tem_palavrao  -> reproduz os 17,1% com palavrao, os 77,4% de toxicas sem
                   palavrao, os 11,6% de nao-toxicas com palavrao e o F1 macro
                   de 0,50 da regra ingenua

Ambas usam exatamente as mesmas definicoes de
app/nlp/evaluation/scripts/analisar_corpus.py, para que os numeros batam.

Uso:
    python -m app.nlp.datasets.gerar_corpus_publico \
        --entrada ../dados-completos/corpus_toxicidade_completo.csv \
        --saida data/corpus_toxicidade_publico.csv
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
    'id_sentenca', 'video_id', 'titulo', 'parte', 'link',
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
                'n_palavras': len(frase.split()),
                'tem_palavrao': int(bool(re.search(PALAVROES, frase.lower()))),
                'tox': r.get('TOX', ''),
                'p1': r.get('P1', ''),
                'p2': r.get('P2', ''),
                'a1': r.get('A1', ''),
            })

    print(f'{len(linhas)} sentencas -> {saida}')
    print('colunas publicadas:', ', '.join(COLUNAS_SAIDA))
    print('a coluna FRASE NAO foi copiada')


def main():
    p = argparse.ArgumentParser(description='Gera a versao publica do corpus')
    p.add_argument('--entrada', required=True,
                   help='CSV completo, com a coluna FRASE (fora do repositorio publico)')
    p.add_argument('--saida', default='data/corpus_toxicidade_publico.csv')
    a = p.parse_args()
    gerar(a.entrada, a.saida)


if __name__ == '__main__':
    main()
