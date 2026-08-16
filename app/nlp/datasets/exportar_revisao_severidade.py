"""
Exporta as sentencas cuja classe conflita com a rubrica Leve/Severa do artigo.

POR QUE ESTE SCRIPT EXISTE
--------------------------
A rubrica publicada (Secao 4.3) define:

  Leve   ofensividade INDIRETA -- critica a objetos, situacoes ou entidades
         abstratas, sem alvo pessoal
  Severa ofensividade DIRETA -- insulto, ameaca ou discurso de odio dirigido
         explicitamente a uma pessoa ou grupo

Na revisao de 2026-08 a fronteira entre as duas deslocou-se: 28 sentencas com
vocativo e insulto em segunda pessoa sairam de Severa para Leve, e o corpus passou
a conter pares quase identicos em classes diferentes. Este script levanta os casos
para uma nova rodada de anotacao, em vez de deixar a incoerencia entrar no artigo.

O QUE ENTRA NA LISTA
--------------------
  dirigida_nao_severa   tem marcador de ofensa dirigida (vocativo do tipo "cala a
                        boca" ou "seu/sua <insulto>") e NAO esta em Severa -- e o
                        conflito direto com a rubrica
  par_divergente        tem uma quase-identica no corpus (cosseno de n-gramas de
                        caractere >= 0.75) com classe diferente
  mudou_na_revisao      o rotulo difere do corpus anterior; e o conjunto que a
                        revisao efetivamente tocou, e vale reconferir inteiro

Um quarto criterio -- "Severa sem marcador de ofensa dirigida" -- foi testado e
descartado: pega 260 sentencas, porque o marcador lexical nao cobre as muitas formas
de dirigir uma ofensa. Revisar 260 de 356 Severas nao e revisao, e reanotacao.

Cada linha traz o rotulo anterior, o atual, as tres anotacoes individuais e uma
coluna TOX_REVISADO em branco para o anotador preencher.

Uso:
    python -m app.nlp.datasets.exportar_revisao_severidade
    python -m app.nlp.datasets.exportar_revisao_severidade --saida ../documentos/revisao.csv
"""

import argparse
import csv
import os
import re
import sys
import unicodedata
from collections import defaultdict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

ARTIGO = os.path.dirname(project_root)
CORPUS_ATUAL = os.path.join(ARTIGO, 'dados-completos', 'corpus_toxicidade_completo.csv')
CORPUS_ANTERIOR = os.path.join(
    os.path.dirname(ARTIGO), '2026--jisa--youtubesafekids-toxicidade',
    'dados-completos', 'corpus_toxicidade_completo.csv')

INSULTOS = (r'(babaca|idiota|imbecil|burr\w+|otari\w+|cretino|nojent\w+|inutil|'
            r'estupid\w+|vadia|corno|arrombad\w+|desgracad\w+|miseravel|fedorent\w+|'
            r'escrot\w+|verme|lixo|animal|monstro|palhac\w+)')
DIRIGIDA = re.compile(r'\b(seu|sua|voce e um|voce e uma|tu es)\s+\w*\s*' + INSULTOS)
VOCATIVO = re.compile(r'\b(cala a boca|cala essa boca|cala esse focinho|cale a boca)\b')
PALAVRAO = re.compile(r'\b(merda|porra|caralho|foda|bosta|puta|buceta|viado|cu|'
                      r'cacete|desgraca)\w*\b')

LIMIAR_PAR = 0.75


def normaliza(t: str) -> str:
    t = unicodedata.normalize('NFKD', t or '')
    t = ''.join(c for c in t if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', '', t.lower())).strip()


def le(caminho: str) -> list:
    with open(caminho, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f, delimiter=';'))


def ngramas(t: str, n: int = 4) -> set:
    t = normaliza(t)
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def pares_divergentes(linhas: list) -> dict:
    """Sentencas com uma quase-identica de classe diferente, por cosseno de n-gramas.

    Blocagem por trigrama compartilhado: comparar 1.506 x 1.506 direto e desperdicio,
    e duas frases sem nenhum trigrama em comum nunca passam do limiar.
    """
    grams = [ngramas(r['FRASE']) for r in linhas]
    indice = defaultdict(list)
    for i, g in enumerate(grams):
        for tri in list(g)[:40]:
            indice[tri].append(i)

    achados = {}
    for i, gi in enumerate(grams):
        candidatos = {j for tri in list(gi)[:40] for j in indice[tri] if j != i}
        for j in candidatos:
            if linhas[i]['TOX'] == linhas[j]['TOX']:
                continue
            gj = grams[j]
            inter = len(gi & gj)
            if not inter:
                continue
            sim = inter / (len(gi) ** 0.5 * len(gj) ** 0.5)
            if sim >= LIMIAR_PAR:
                anterior = achados.get(i)
                if not anterior or sim > anterior[1]:
                    achados[i] = (j, sim)
    return achados


def main():
    ap = argparse.ArgumentParser(description='Casos para revisao da fronteira Leve/Severa')
    ap.add_argument('--saida', default=os.path.join(
        ARTIGO, 'documentos', 'revisao-fronteira-leve-severa.csv'))
    args = ap.parse_args()

    atual = le(CORPUS_ATUAL)
    try:
        anterior = {normaliza(r['FRASE']): r['TOX'] for r in le(CORPUS_ANTERIOR)}
    except FileNotFoundError:
        anterior = {}
        print(f"aviso: corpus anterior nao encontrado em {CORPUS_ANTERIOR}")

    pares = pares_divergentes(atual)

    linhas = []
    for i, r in enumerate(atual):
        f = normaliza(r['FRASE'])
        motivos = []
        tem_marcador = bool(DIRIGIDA.search(f) or VOCATIVO.search(f))

        if tem_marcador and r['TOX'] != 'Severa':
            motivos.append('dirigida_nao_severa')
        if i in pares:
            motivos.append('par_divergente')
        antes = anterior.get(f)
        if antes and antes != r['TOX']:
            motivos.append('mudou_na_revisao')
        if not motivos:
            continue

        j, sim = pares.get(i, (None, None))
        linhas.append({
            'id_sentenca': i + 1,
            'video_id': r['ID'],
            'titulo': r['TITULO'],
            'parte': r['PARTE'],
            'frase': r['FRASE'],
            'tox_anterior': anterior.get(f, '(nao estava no corpus anterior)'),
            'tox_atual': r['TOX'],
            'p1': r['P1'], 'p2': r['P2'], 'a1': r['A1'],
            'unanime': 'sim' if r['P1'] == r['P2'] == r['A1'] else 'nao',
            'motivo': ' + '.join(motivos),
            'par_frase': atual[j]['FRASE'] if j is not None else '',
            'par_classe': atual[j]['TOX'] if j is not None else '',
            'par_similaridade': f"{sim:.2f}" if sim else '',
            'TOX_REVISADO': '',
            'COMENTARIO': '',
        })

    linhas.sort(key=lambda d: (d['motivo'], d['tox_atual'], d['frase']))

    os.makedirs(os.path.dirname(args.saida), exist_ok=True)
    with open(args.saida, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()), delimiter=';')
        w.writeheader()
        w.writerows(linhas)

    print(f"{len(linhas)} sentencas para revisao -> {args.saida}\n")
    por_motivo = defaultdict(int)
    for l in linhas:
        por_motivo[l['motivo']] += 1
    for m, n in sorted(por_motivo.items(), key=lambda x: -x[1]):
        print(f"  {n:>4}  {m}")
    print()
    migrou = sum(1 for l in linhas
                 if l['tox_anterior'] == 'Severa' and l['tox_atual'] == 'Leve')
    print(f"  {migrou} destas sairam de Severa para Leve na revisao de 2026-08")


if __name__ == '__main__':
    main()
