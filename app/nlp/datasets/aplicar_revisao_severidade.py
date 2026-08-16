"""
Aplica ao corpus as reanotacoes da revisao da fronteira Leve/Severa.

ENTRADA
-------
documentos/revisao-fronteira-leve-severa_rotulado.csv -- a planilha exportada por
exportar_revisao_severidade.py, devolvida pelos anotadores com as colunas
`tox_atual`, `p1`, `p2` e `a1` atualizadas.

O QUE FAZ
---------
Casa cada linha pelo `id_sentenca` (posicao 1-based no corpus) e CONFERE a frase antes
de escrever: se o texto nao bater, aborta em vez de corromper o corpus silenciosamente.
Grava as tres anotacoes individuais e o rotulo final.

REGRA DO ROTULO FINAL
---------------------
A Secao 4.3 do artigo declara que o rotulo final e a MODA das tres anotacoes. Quando a
coluna `tox_atual` diverge da moda de `p1`/`p2`/`a1`, prevalece a moda e a divergencia e
reportada -- manter a excecao quebraria a regra que o artigo publica e que qualquer leitor
do corpus publico pode recalcular. Sem maioria (tres rotulos distintos), mantem-se o que a
planilha traz em `tox_atual`.

Uso:
    python -m app.nlp.datasets.aplicar_revisao_severidade
    python -m app.nlp.datasets.aplicar_revisao_severidade --simular   # so relata
"""

import argparse
import csv
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

ARTIGO = os.path.dirname(project_root)
CORPUS = os.path.join(ARTIGO, 'dados-completos', 'corpus_toxicidade_completo.csv')
REVISAO = os.path.join(ARTIGO, 'documentos',
                       'revisao-fronteira-leve-severa_rotulado.csv')
CLASSES = {'Nenhuma', 'Leve', 'Severa'}


def le(caminho, delim=';'):
    with open(caminho, encoding='utf-8-sig') as f:
        r = csv.DictReader(f, delimiter=delim)
        return list(r), r.fieldnames


def main():
    ap = argparse.ArgumentParser(description='Aplica a revisao ao corpus')
    ap.add_argument('--simular', action='store_true',
                    help='relata as mudancas sem escrever nada')
    args = ap.parse_args()

    corpus, campos = le(CORPUS)
    revisao, _ = le(REVISAO)
    print(f"corpus: {len(corpus)} sentencas | revisao: {len(revisao)} linhas\n")

    # --- validacao antes de tocar em qualquer coisa -------------------------
    erros = []
    for r in revisao:
        try:
            i = int(r['id_sentenca']) - 1
        except ValueError:
            erros.append(f"id_sentenca nao numerico: {r['id_sentenca']!r}")
            continue
        if not 0 <= i < len(corpus):
            erros.append(f"id {r['id_sentenca']} fora do corpus")
        elif corpus[i]['FRASE'].strip() != r['frase'].strip():
            erros.append(f"id {r['id_sentenca']}: frase nao confere\n"
                         f"      corpus:  {corpus[i]['FRASE'][:70]!r}\n"
                         f"      revisao: {r['frase'][:70]!r}")
        for c in ('tox_atual', 'p1', 'p2', 'a1'):
            if r[c] not in CLASSES:
                erros.append(f"id {r['id_sentenca']}: {c}={r[c]!r} nao e uma classe")

    if erros:
        print("ABORTADO -- a planilha nao passou na validacao:\n")
        for e in erros[:20]:
            print(f"  {e}")
        if len(erros) > 20:
            print(f"  ... e mais {len(erros) - 20}")
        return 1

    # --- aplicacao ----------------------------------------------------------
    migracoes = Counter()
    por_anotador = Counter()
    corrigidos_pela_moda = []
    alterados = 0

    for r in revisao:
        i = int(r['id_sentenca']) - 1
        linha = corpus[i]

        moda, n = Counter([r['p1'], r['p2'], r['a1']]).most_common(1)[0]
        final = moda if n >= 2 else r['tox_atual']
        if n >= 2 and moda != r['tox_atual']:
            corrigidos_pela_moda.append(
                (r['id_sentenca'], r['tox_atual'], moda,
                 f"{r['p1']}/{r['p2']}/{r['a1']}", r['frase'][:56]))

        antes = linha['TOX']
        for coluna, valor in (('P1', r['p1']), ('P2', r['p2']), ('A1', r['a1'])):
            if linha[coluna] != valor:
                por_anotador[coluna] += 1
            linha[coluna] = valor
        linha['TOX'] = final

        if antes != final:
            migracoes[(antes, final)] += 1
            alterados += 1

    # --- relatorio ----------------------------------------------------------
    if corrigidos_pela_moda:
        print(f"{len(corrigidos_pela_moda)} linha(s) com `tox_atual` divergente da moda "
              f"-- prevaleceu a moda, conforme a Secao 4.3:\n")
        for sid, planilha, moda, anots, frase in corrigidos_pela_moda:
            print(f"  id {sid}: planilha={planilha} anotacoes={anots} -> {moda}")
            print(f"          {frase}")
        print()

    print("anotacoes individuais alteradas:")
    for c in ('P1', 'P2', 'A1'):
        print(f"  {c}: {por_anotador[c]}")
    print()

    print(f"rotulo final alterado em {alterados} de {len(revisao)} sentencas revisadas:")
    for (a, b), n in sorted(migracoes.items(), key=lambda x: -x[1]):
        print(f"  {a:<8} -> {b:<8} {n:>4}")
    print()

    dist = Counter(l['TOX'] for l in corpus)
    print("distribuicao do corpus depois da revisao:")
    for c in ('Nenhuma', 'Leve', 'Severa'):
        print(f"  {c:<8} {dist[c]:>5}  ({100*dist[c]/len(corpus):.2f}%)")

    if args.simular:
        print("\n[--simular] nada foi escrito.")
        return 0

    backup = CORPUS.replace('.csv', f"_pre-revisao_{datetime.now():%Y%m%d_%H%M%S}.csv")
    shutil.copy2(CORPUS, backup)
    with open(CORPUS, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=campos, delimiter=';')
        w.writeheader()
        w.writerows(corpus)

    print(f"\ncorpus gravado: {CORPUS}")
    print(f"backup:         {os.path.basename(backup)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
