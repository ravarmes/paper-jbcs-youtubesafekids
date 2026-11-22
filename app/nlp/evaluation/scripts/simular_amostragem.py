"""
Simulação de amostragem: quanto da detecção no nível do vídeo depende de quantas
frases são amostradas e de como elas são agregadas?

Um estudo de caso com quatro buscas mostra que um fenômeno ocorre; não mede sua
magnitude. Esta simulação mede, sob hipóteses declaradas, a taxa de detecção de um
vídeo com conteúdo severamente tóxico quando o serviço lê só k frases dele, com o
limiar calibrado para 5% de falso alarme em vídeos inteiramente apropriados, de
modo que todas as regras e todos os k sejam comparados no mesmo ponto de operação.
Atende aos itens M5a, M5b, M5c e M7c do plano de resposta aos revisores
(documentos/plano-resposta-revisores.md, pacote P8) e fornece ao estudo de caso o
limiar fixado a priori (M4b).

SCORE DE UMA FRASE
------------------
P = 1 - (0,5*p_leve + 1,0*p_severa), com as probabilidades do classificador; o
score de um vídeo é uma regra de agregação dos k scores de frase (REGRAS abaixo;
'media' é a Equação do artigo).

POPULAÇÕES DE AMOSTRAGEM (as frases de que os vídeos sintéticos são feitos)
---------------------------------------------------------------------------
  'oof'    predições out-of-fold sobre a partição de treino, gravadas pelo treino
           final em results/<tag>/modelo_final/probabilidades_oof_<ts>.csv (cada
           frase predita pelo modelo do fold que não a viu, folds agrupados por
           vídeo). É a população que pode fixar limiares A PRIORI: não toca no
           teste nem no estudo de caso.
  'teste'  o teste retido, com as probabilidades do modelo de referência
           (probabilidades_teste_<ts>.csv, colunas da semente de referência). Se o
           CSV não existir, recai na inferência do modelo de referência sobre o
           teste (comportamento da versão submetida).

HIPÓTESES (declaradas também no JSON, bloco 'hipoteses')
--------------------------------------------------------
  M5c  Frases apropriadas: na variante de referência ('iid_nenhuma') a parte
       apropriada do vídeo é sorteada SÓ entre as frases Nenhuma; frases Leve não
       entram em vídeo nenhum. O conteúdo tóxico é sempre de frases Severa (a
       detecção mede a presença de Severas). A variante 'iid_nenhuma_leve' sorteia a
       parte apropriada do conjunto Nenhuma ∪ Leve (uniforme sobre as frases, isto
       é, na proporção em que as duas classes aparecem na população): Leve passa a
       contar como conteúdo apropriado, não-alvo, e entra também na calibração do
       limiar.
  M5a  Independência: nas variantes 'iid_*' as k frases são sorteadas de forma
       independente e cada uma é Severa com probabilidade f, sem estrutura
       temporal. As variantes 'bloco_*' relaxam isso: o vídeo é uma sequência
       ordenada de L = 200 frases cujo conteúdo tóxico é UM segmento contíguo de
       round(f*L) frases Severas, com início sorteado uniformemente entre as
       posições em que cabe inteiro; o serviço lê k posições:
         'bloco_fixo'           posições fixas e igualmente espaçadas,
                                round(linspace(0, L-1, k)) -- para k = 3: primeira,
                                do meio e última frase, como o serviço (início, meio
                                e fim; ver app/core/youtube.py);
         'bloco_estratificado'  uma posição sorteada uniformemente em cada um de k
                                estratos iguais do vídeo.
       f varia em FRACOES = 0,05; 0,10; 0,20; 0,30.

INCERTEZA (M5b)
---------------
Cada célula traz a detecção com uma casa decimal e dois intervalos de 95%: o de
Wilson sobre os N_SIM vídeos simulados (erro de Monte Carlo) e o percentil 2,5-97,5
de N_BOOT_POP reamostragens bootstrap da população de frases (estratificadas por
classe; cada uma recalibra o limiar e refaz a simulação com N_SIM_BOOT vídeos),
que carrega a incerteza de a população ter um número finito de frases.

DIAGNÓSTICO (M5b)
-----------------
Bloco 'diagnostico_m5b', por população, na variante de referência com f = 0,20:
distribuição dos scores de frase por classe (bimodalidade), limiares calibrados,
distribuição do número de Severas amostradas, detecção condicional a esse número
para a média e a média das 2 piores, concordância entre as duas regras e entre
cada regra e a regra de contagem "ao menos c Severas". Explica (a) a queda da
detecção pela média de k = 3 para k = 5 e (b) por que 'duas_piores' ≈ 'media'.

LIMIAR A PRIORI (M4b) E BANDA DE TOLERÂNCIA (M7c)
-------------------------------------------------
  'limiar_a_priori'   percentil 5 do score (média das k = 3) em vídeos sintéticos
                      inteiramente apropriados (só Nenhuma), população OOF. É o
                      limiar que analisar_estratos.py usa como designado.
  'banda_tolerancia'  delta = percentil 95 de |S1 - S2| entre dois vídeos
                      sintéticos inteiramente apropriados e independentes (k = 3,
                      média, população OOF). Regra de reordenação que o usa:
                      analisar_reordenacao.ordenar_com_banda.
  Se as OOF não existirem, ambos recaem na população de teste, com o campo
  'populacao' e um 'aviso' dizendo isso.

Cada célula usa um gerador próprio, semeado por (SEMENTE, população, variante,
f, k): os números não dependem da ordem dos laços nem de células acrescentadas.

SAÍDAS
------
  <resultados>/simulacao_amostragem_<ts>.json
  dir_latex('tabelas')/tab-simulacao.tex            (ou <saida>/latex/tabelas/)
  dir_latex('tabelas')/tab-simulacao-variantes.tex

Uso (de src/):
    python -m app.nlp.evaluation.scripts.simular_amostragem
    # validação com as entradas de outra rodada, saída noutra pasta:
    python -m app.nlp.evaluation.scripts.simular_amostragem \
        --resultados app/nlp/evaluation/results/video-original --saida <pasta>
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp import experimento  # noqa: E402

logging.disable(logging.INFO)


def importar_irmao(nome):
    """
    Importa um script desta pasta sem passar por app.nlp.evaluation, cujo
    __init__ carrega o avaliador de modelos (e, com ele, torch). Assim a
    simulação a partir dos CSV de probabilidades roda sem torch quando chamada
    pelo caminho do arquivo (python app/nlp/evaluation/scripts/<script>.py).
    """
    import importlib
    pasta = os.path.dirname(os.path.abspath(__file__))
    if pasta not in sys.path:
        sys.path.insert(0, pasta)
    return importlib.import_module(nome)


PESOS = {'Nenhuma': 0.0, 'Leve': 0.5, 'Severa': 1.0}
COLS_P = ('p_nenhuma', 'p_leve', 'p_severa')
SEED = 7
N_SIM = 30000
KS = (3, 5, 8, 12)
K_SERVICO = 3                    # frases que o serviço lê hoje (início, meio, fim)
FRACOES = (0.05, 0.10, 0.20, 0.30)
F_TABELA = 0.20                  # fração da tabela principal e do diagnóstico
FALSO_ALARME_ALVO = 5            # percentil, em vídeo 100% apropriado
L_VIDEO = 200                    # frases de um vídeo ordenado (variantes de bloco)
N_BOOT_POP = 200                 # reamostragens bootstrap da população de frases
N_SIM_BOOT = 4000                # vídeos simulados por célula em cada reamostragem
PERCENTIL_BANDA = 95             # percentil de |ΔS| que define a banda de tolerância
QUANTIS_DIF = (50, 75, 90, 95, 99)

REGRAS = {
    'media': lambda s: s.mean(axis=1),
    'minimo': lambda s: s.min(axis=1),
    'min_ponderado': lambda s: 0.7 * s.min(axis=1) + 0.3 * s.mean(axis=1),
    'duas_piores': lambda s: np.sort(s, axis=1)[:, :2].mean(axis=1),
}
DESCRICAO_REGRAS = {
    'media': 'média dos k scores de frase (Equação do artigo)',
    'minimo': 'menor dos k scores de frase',
    'min_ponderado': '0,7 * mínimo + 0,3 * média',
    'duas_piores': 'média dos 2 menores scores de frase',
}

# base: de onde vem a parte apropriada do vídeo; estrutura: iid ou bloco contíguo
VARIANTES = {
    'iid_nenhuma': {'base': 'nenhuma', 'estrutura': 'iid'},
    'iid_nenhuma_leve': {'base': 'nenhuma_leve', 'estrutura': 'iid'},
    'bloco_fixo': {'base': 'nenhuma', 'estrutura': 'bloco', 'posicoes': 'fixas'},
    'bloco_estratificado': {'base': 'nenhuma', 'estrutura': 'bloco',
                            'posicoes': 'estratificadas'},
}

# Códigos para semear o gerador de cada célula
COD_POP = {'oof': 1, 'teste': 2}
COD_BASE = {'nenhuma': 1, 'nenhuma_leve': 2}
COD_VAR = {'iid_nenhuma': 11, 'iid_nenhuma_leve': 12, 'bloco_fixo': 13,
           'bloco_estratificado': 14}
COD_CALIBRACAO, COD_DIAGNOSTICO, COD_DISPERSAO, COD_BOOTSTRAP = 1, 21, 31, 51

HIPOTESES = {
    'frases_apropriadas': (
        "Variante de referência 'iid_nenhuma': a parte apropriada do vídeo é sorteada só "
        "entre as frases Nenhuma da população; frases Leve não entram em vídeo nenhum. "
        "Variante 'iid_nenhuma_leve': a parte apropriada é sorteada uniformemente no "
        "conjunto Nenhuma ∪ Leve (proporção das classes na população; ver "
        "populacoes.<p>.fracao_leve_na_parte_apropriada)."),
    'tratamento_leve': (
        "Leve nunca é conteúdo-alvo: o alvo da detecção é a presença de frases Severa. Na "
        "variante 'iid_nenhuma_leve', Leve conta como conteúdo apropriado e entra também na "
        "calibração do limiar (os vídeos inteiramente apropriados misturam Nenhuma e Leve). "
        "No score, Leve pesa 0,5."),
    'frases_toxicas': 'O conteúdo tóxico é sempre de frases Severa da população.',
    'independencia': (
        "Variantes 'iid_*': as k frases amostradas são sorteadas de forma independente, cada "
        "uma Severa com probabilidade f, sem estrutura temporal nem continuidade de tema "
        "(hipótese da versão submetida). Vídeos reais têm agrupamento temporal; as variantes "
        "'bloco_*' o modelam."),
    'bloco_contiguo': {
        'L_frases_por_video': L_VIDEO,
        'frases_toxicas_no_bloco': 'max(1, round(f * L))',
        'inicio_do_bloco': 'uniforme entre 0 e L - round(f*L) (o bloco cabe inteiro no vídeo)',
        'posicoes_fixas': ('round(linspace(0, L-1, k)): igualmente espaçadas, incluindo a '
                           'primeira e a última frase; para k=3, início, meio e fim'),
        'posicoes_estratificadas': 'uma posição uniforme em cada um de k estratos iguais',
        'parte_apropriada': 'frases Nenhuma, sorteadas de forma independente',
    },
    'calibracao': (
        'Cada regra e cada k têm seu limiar: o percentil 5 do score em N_SIM vídeos '
        'sintéticos inteiramente apropriados da mesma população e com a mesma base '
        '(Nenhuma, ou Nenhuma ∪ Leve). O vídeo é detectado se score < limiar. O falso '
        'alarme é conferido numa amostra independente (falso_alarme_verificado).'),
}


# =============================================================================
# Entradas
# =============================================================================

def localizar_modelo_final(pasta):
    """
    Arquivos do treino final com um mesmo timestamp (o mais recente) em
    <pasta>/modelo_final/. Devolve dict com os caminhos (None quando faltar) ou
    None se a pasta não existir.
    """
    pasta_mf = Path(pasta) / 'modelo_final'
    if not pasta_mf.is_dir():
        return None
    jsons = sorted(pasta_mf.glob('modelo_final_*.json'))
    if jsons:
        ts = jsons[-1].stem[len('modelo_final_'):]
    else:
        oofs = sorted(pasta_mf.glob('probabilidades_oof_*.csv'))
        if not oofs:
            return None
        ts = oofs[-1].stem[len('probabilidades_oof_'):]
    arq = {'ts': ts,
           'oof': pasta_mf / f'probabilidades_oof_{ts}.csv',
           'teste': pasta_mf / f'probabilidades_teste_{ts}.csv',
           'json': pasta_mf / f'modelo_final_{ts}.json'}
    return {k: (v if isinstance(v, str) or v.exists() else None) for k, v in arq.items()}


def _normalizar(P):
    # O CSV traz ~8 algarismos (float32); renormaliza cada linha para somar 1.
    P = np.asarray(P, dtype=float)
    return P / P.sum(axis=1, keepdims=True)


def score_frases(P, w_leve=PESOS['Leve']):
    """Score de segurança de cada frase: 1 - (w_leve*p_leve + 1,0*p_severa)."""
    P = np.asarray(P, dtype=float)
    return 1.0 - (w_leve * P[:, 1] + PESOS['Severa'] * P[:, 2])


def montar_populacao(nome, P, y, origem, descricao):
    P = _normalizar(P)
    y = np.asarray(y, dtype=int)
    idx = {c: np.where(y == c)[0] for c in (0, 1, 2)}
    n_apr = len(idx[0]) + len(idx[1])
    return {
        'nome': nome, 'P': P, 'y': y, 'scores': score_frases(P), 'idx': idx,
        'info': {
            'origem': str(origem), 'descricao': descricao, 'n': int(len(y)),
            'n_por_classe': {'Nenhuma': int(len(idx[0])), 'Leve': int(len(idx[1])),
                             'Severa': int(len(idx[2]))},
            'fracao_leve_na_parte_apropriada': (len(idx[1]) / n_apr) if n_apr else None,
        },
    }


def carregar_populacao_oof(arquivos):
    import pandas as pd

    if not arquivos or not arquivos.get('oof'):
        return None
    df = pd.read_csv(arquivos['oof'], sep=';', dtype={'video_id': str})
    validas = df[list(COLS_P)].notna().all(axis=1)
    df = df[validas].reset_index(drop=True)
    pop = montar_populacao('oof', df[list(COLS_P)].to_numpy(), df['rotulo'].to_numpy(),
                           arquivos['oof'],
                           'predições out-of-fold sobre a partição de treino '
                           '(folds agrupados por vídeo)')
    pop['info']['descartadas_sem_predicao'] = int((~validas).sum())
    return pop


def _semente_referencia(arquivos, sementes):
    if arquivos and arquivos.get('json'):
        try:
            with open(arquivos['json'], encoding='utf-8') as fh:
                ref = json.load(fh).get('semente_referencia')
            if ref in sementes:
                return int(ref)
        except (OSError, ValueError):
            pass
    return sementes[0]


def carregar_populacao_teste(arquivos, modelo_override=None):
    """Teste retido: do CSV do treino final ou, na falta dele, por inferência."""
    import pandas as pd

    if arquivos and arquivos.get('teste'):
        df = pd.read_csv(arquivos['teste'], sep=';')
        sementes = [int(m.group(1)) for c in df.columns
                    for m in [re.fullmatch(r'p_nenhuma_s(\d+)', c)] if m]
        if sementes:
            ref = _semente_referencia(arquivos, sementes)
            P = df[[f'{c}_s{ref}' for c in COLS_P]].to_numpy()
            pop = montar_populacao('teste', P, df['rotulo'].to_numpy(), arquivos['teste'],
                                   f'teste retido, probabilidades do modelo de referência '
                                   f'(semente {ref})')
            pop['info']['semente_referencia'] = ref
            return pop

    # Recurso: inferência do modelo de referência sobre o teste (versão submetida)
    from app.nlp.datasets.prepare_data_toxicity import get_data_for_cv_and_test
    from app.nlp.models.bertimbau_toxicity import BertimbauToxicity

    caminho = Path(modelo_override) if modelo_override else experimento.caminho_modelo_referencia()
    print(f'Modelo (inferência no teste): {caminho.name}')
    modelo = BertimbauToxicity(model_path=str(caminho))
    (_, _), (X_test, y_test) = get_data_for_cv_and_test()
    P = []
    for t in X_test:
        probs = modelo.predict_toxicity(t, return_probabilities=True)['probabilities']
        P.append([probs['Nenhuma'], probs['Leve'], probs['Severa']])
    return montar_populacao('teste', np.array(P), y_test, caminho,
                            'teste retido, inferência do modelo de referência')


def carregar_populacoes(pasta_resultados, modelo_override=None):
    arquivos = localizar_modelo_final(pasta_resultados)
    pops = {}
    oof = carregar_populacao_oof(arquivos)
    if oof is not None:
        pops['oof'] = oof
    pops['teste'] = carregar_populacao_teste(arquivos, modelo_override)
    return pops, arquivos


# =============================================================================
# Sorteio dos vídeos sintéticos
# =============================================================================

def _rng(*codigos):
    return np.random.default_rng([SEED] + [int(c) for c in codigos])


def idx_apropriadas(idx, base):
    if base == 'nenhuma':
        return idx[0]
    return np.concatenate([idx[0], idx[1]])


def videos_apropriados(rng, scores, idx_apr, k, n):
    """n vídeos inteiramente apropriados: k frases sorteadas de idx_apr."""
    return scores[rng.choice(idx_apr, size=(n, k))]


def n_toxicas_bloco(f, L=L_VIDEO):
    return 0 if f <= 0 else max(1, int(round(f * L)))


def posicoes_amostradas(rng, modo, k, n, L=L_VIDEO):
    """Posições (0..L-1) que o serviço lê em cada um dos n vídeos."""
    if modo == 'fixas':
        pos = np.round(np.linspace(0, L - 1, k)).astype(int)
        return np.broadcast_to(pos, (n, k))
    bordas = np.floor(np.linspace(0, L, k + 1)).astype(int)
    lo, largura = bordas[:-1], np.diff(bordas)
    return lo + (rng.random((n, k)) * largura).astype(int)


def sortear_videos(rng, scores, idx_apr, idx_sev, variante, k, f, n):
    """
    n vídeos com conteúdo severamente tóxico na fração f, segundo a variante.
    Devolve (scores k por vídeo, máscara das posições amostradas que são Severas).
    """
    cfg = VARIANTES[variante]
    if f <= 0:
        tox = np.zeros((n, k), dtype=bool)
    elif cfg['estrutura'] == 'iid':
        tox = rng.random((n, k)) < f
    else:
        m = n_toxicas_bloco(f)
        pos = posicoes_amostradas(rng, cfg['posicoes'], k, n)
        ini = rng.integers(0, L_VIDEO - m + 1, size=(n, 1))
        tox = (pos >= ini) & (pos < ini + m)
    s = scores[rng.choice(idx_apr, size=(n, k))]
    if tox.any():
        s = np.where(tox, scores[rng.choice(idx_sev, size=(n, k))], s)
    return s, tox


# =============================================================================
# Estatística
# =============================================================================

def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    meia = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(centro - meia), float(centro + meia)]


def _quantis(x, ps=(5, 25, 50, 75, 95)):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {f'p{p}': None for p in ps}
    return {f'p{p}': float(np.percentile(x, p)) for p in ps}


def _ic_percentil(valores):
    v = np.asarray([x for x in valores if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return [None, None]
    lo, hi = np.percentile(v, [2.5, 97.5])
    return [float(lo), float(hi)]


def distribuicao_scores(pop):
    """Distribuição dos scores de frase por classe: bimodalidade (M5b)."""
    s, y = pop['scores'], pop['y']
    bordas = np.round(np.linspace(0, 1, 11), 2)
    out = {'bordas_histograma': bordas.tolist(), 'por_classe': {}}
    for c, nome in enumerate(('Nenhuma', 'Leve', 'Severa')):
        sc = s[y == c]
        hist = np.histogram(sc, bins=bordas)[0] if sc.size else np.zeros(10)
        out['por_classe'][nome] = {
            'n': int(sc.size), 'quantis': _quantis(sc), 'media': float(sc.mean()) if sc.size else None,
            'fracao_por_faixa': (hist / max(sc.size, 1)).tolist(),
            'fracao_abaixo_0_5': float((sc < 0.5).mean()) if sc.size else None,
            'fracao_abaixo_0_1': float((sc < 0.1).mean()) if sc.size else None,
            'fracao_acima_0_9': float((sc > 0.9).mean()) if sc.size else None,
        }
    out['todas'] = {
        'fracao_extremos_abaixo_0_1_ou_acima_0_9': float(((s < 0.1) | (s > 0.9)).mean()),
        'fracao_faixa_leve_0_4_a_0_6': float(((s >= 0.4) & (s <= 0.6)).mean()),
        'fracao_intermediaria_0_1_a_0_4_ou_0_6_a_0_9': float(
            (((s >= 0.1) & (s < 0.4)) | ((s > 0.6) & (s <= 0.9))).mean()),
    }
    return out


# =============================================================================
# Simulação
# =============================================================================

def calibrar_celula(pop, base, k, n=None):
    """
    Limiar de cada regra para (população, base, k): percentil FALSO_ALARME_ALVO do
    score em n vídeos inteiramente apropriados; falso alarme conferido numa amostra
    independente. O gerador é semeado só por (população, base, k), de modo que
    analisar_estratos.py reproduz o limiar a priori sem refazer a simulação.
    """
    n = n or N_SIM
    rng = _rng(COD_CALIBRACAO, COD_POP[pop['nome']], COD_BASE[base], k)
    idx_apr = idx_apropriadas(pop['idx'], base)
    s = videos_apropriados(rng, pop['scores'], idx_apr, k, n)
    s_ver = videos_apropriados(rng, pop['scores'], idx_apr, k, n)
    out = {}
    for regra, f_r in REGRAS.items():
        lim = float(np.percentile(f_r(s), FALSO_ALARME_ALVO))
        out[regra] = {'limiar': lim, 'falso_alarme_verificado': float((f_r(s_ver) < lim).mean())}
    return out


def simular_populacao(pop):
    """Calibração e detecção em todas as células (variante x f x k x regra)."""
    limiares = {base: {regra: {} for regra in REGRAS} for base in COD_BASE}
    for base in COD_BASE:
        for k in KS:
            for regra, d in calibrar_celula(pop, base, k).items():
                limiares[base][regra][str(k)] = d

    deteccao, amostragem = {}, {}
    for variante, cfg in VARIANTES.items():
        deteccao[variante], amostragem[variante] = {}, {}
        idx_apr = idx_apropriadas(pop['idx'], cfg['base'])
        for f in FRACOES:
            fk = f'{f:.2f}'
            deteccao[variante][fk] = {regra: {} for regra in REGRAS}
            amostragem[variante][fk] = {}
            for k in KS:
                rng = _rng(COD_POP[pop['nome']], COD_VAR[variante], round(f * 100), k)
                s, tox = sortear_videos(rng, pop['scores'], idx_apr, pop['idx'][2],
                                        variante, k, f, N_SIM)
                n_sev = tox.sum(axis=1)
                amostragem[variante][fk][str(k)] = {
                    'p_ao_menos_uma_severa_amostrada': float((n_sev >= 1).mean()),
                    'media_severas_amostradas': float(n_sev.mean()),
                }
                for regra, f_r in REGRAS.items():
                    lim = limiares[cfg['base']][regra][str(k)]['limiar']
                    acertos = int((f_r(s) < lim).sum())
                    p = acertos / N_SIM
                    deteccao[variante][fk][regra][str(k)] = {
                        'limiar': lim, 'deteccao': p, 'deteccao_pct': round(100 * p, 1),
                        'ic95_wilson_mc': wilson(acertos, N_SIM),
                    }
    return {'limiares': limiares, 'deteccao': deteccao, 'amostragem': amostragem}


def bootstrap_populacao(pop, res, n_boot=None, n_sim=None):
    """
    IC 95% de cada célula por bootstrap da população de frases: reamostra as frases
    de cada classe com reposição, recalibra os limiares e refaz a detecção. Grava
    os intervalos dentro de `res` (chaves 'ic95_bootstrap_populacao').
    """
    n_boot = N_BOOT_POP if n_boot is None else n_boot
    n_sim = N_SIM_BOOT if n_sim is None else n_sim
    rng = _rng(COD_BOOTSTRAP, COD_POP[pop['nome']])
    scores = pop['scores']
    acum_det = {}
    acum_lim = {}
    acum_banda = []
    acum_lim_apriori = []
    for _ in range(n_boot):
        idx_b = {c: (rng.choice(pop['idx'][c], size=len(pop['idx'][c]))
                     if len(pop['idx'][c]) else pop['idx'][c]) for c in (0, 1, 2)}
        lim = {}
        for base in COD_BASE:
            idx_apr = idx_apropriadas(idx_b, base)
            for k in KS:
                s = videos_apropriados(rng, scores, idx_apr, k, n_sim)
                for regra, f_r in REGRAS.items():
                    v = float(np.percentile(f_r(s), FALSO_ALARME_ALVO))
                    lim[(base, regra, k)] = v
                    acum_lim.setdefault((base, regra, k), []).append(v)
                if base == 'nenhuma' and k == K_SERVICO:
                    s2 = videos_apropriados(rng, scores, idx_apr, k, n_sim)
                    dif = np.abs(s.mean(axis=1) - s2.mean(axis=1))
                    acum_banda.append(float(np.percentile(dif, PERCENTIL_BANDA)))
                    acum_lim_apriori.append(lim[(base, 'media', k)])
        for variante, cfg in VARIANTES.items():
            idx_apr = idx_apropriadas(idx_b, cfg['base'])
            for f in FRACOES:
                for k in KS:
                    s, _ = sortear_videos(rng, scores, idx_apr, idx_b[2], variante, k, f, n_sim)
                    for regra, f_r in REGRAS.items():
                        det = float((f_r(s) < lim[(cfg['base'], regra, k)]).mean())
                        acum_det.setdefault((variante, f'{f:.2f}', regra, str(k)), []).append(det)

    for (variante, fk, regra, k), vals in acum_det.items():
        res['deteccao'][variante][fk][regra][k]['ic95_bootstrap_populacao'] = _ic_percentil(vals)
    for (base, regra, k), vals in acum_lim.items():
        res['limiares'][base][regra][str(k)]['ic95_bootstrap_populacao'] = _ic_percentil(vals)
    return {'banda_p95_k3': _ic_percentil(acum_banda),
            'limiar_media_k3_nenhuma': _ic_percentil(acum_lim_apriori)}


def diagnostico_m5b(pop, limiares):
    """
    Evidência para as duas anomalias da Tabela 9 (variante iid_nenhuma, f = F_TABELA):
      (a) a detecção pela média cai de k = 3 para k = 5;
      (b) 'duas_piores' quase coincide com 'media' em k = 3, 5 e 8.
    """
    from scipy.stats import spearmanr

    scores, idx = pop['scores'], pop['idx']
    nenhuma = scores[idx[0]]
    por_k = {}
    for k in KS:
        rng = _rng(COD_DIAGNOSTICO, COD_POP[pop['nome']], k)
        s, tox = sortear_videos(rng, scores, idx[0], idx[2], 'iid_nenhuma', k, F_TABELA, N_SIM)
        apr = videos_apropriados(rng, scores, idx[0], k, N_SIM)
        n_sev = tox.sum(axis=1)
        m, d2 = REGRAS['media'](s), REGRAS['duas_piores'](s)
        lm = limiares['nenhuma']['media'][str(k)]['limiar']
        ld = limiares['nenhuma']['duas_piores'][str(k)]['limiar']
        det_m, det_d = m < lm, d2 < ld

        por_numero = {}
        for j in range(k + 1):
            sel = n_sev == j
            p_teo = math.comb(k, j) * F_TABELA ** j * (1 - F_TABELA) ** (k - j)
            if not sel.any() and p_teo < 1e-4:
                continue
            por_numero[str(j)] = {
                'p_simulada': float(sel.mean()), 'p_binomial': float(p_teo),
                'deteccao_media': float(det_m[sel].mean()) if sel.any() else None,
                'deteccao_duas_piores': float(det_d[sel].mean()) if sel.any() else None,
            }

        uma = n_sev == 1
        m_apr, d_apr = REGRAS['media'](apr), REGRAS['duas_piores'](apr)
        dm_apr, dd_apr = m_apr < lm, d_apr < ld
        baixa = apr < 0.5

        def concordancia(a, b):
            return {'concordancia': float((a == b).mean()),
                    'ambas_detectam': float((a & b).mean()),
                    'so_media': float((a & ~b).mean()),
                    'so_duas_piores': float((~a & b).mean()),
                    'nenhuma_detecta': float((~a & ~b).mean())}

        por_k[str(k)] = {
            'limiar_media': lm, 'limiar_duas_piores': ld,
            'deteccao_media': float(det_m.mean()), 'deteccao_duas_piores': float(det_d.mean()),
            'p_ao_menos_uma_severa': float((n_sev >= 1).mean()),
            'p_ao_menos_duas_severas': float((n_sev >= 2).mean()),
            'por_numero_de_severas': por_numero,
            'video_com_exatamente_uma_severa': {
                'fracao_dos_videos': float(uma.mean()),
                'score_media_quantis': _quantis(m[uma]),
                'fracao_abaixo_do_limiar_media': float(det_m[uma].mean()) if uma.any() else None,
                'score_duas_piores_quantis': _quantis(d2[uma]),
                'fracao_abaixo_do_limiar_duas_piores': (float(det_d[uma].mean())
                                                        if uma.any() else None),
            },
            'videos_apropriados': {
                'score_media_quantis': _quantis(m_apr, (1, 5, 10, 50)),
                'fracao_com_ao_menos_uma_frase_abaixo_0_5': float(baixa.any(axis=1).mean()),
                'falso_alarme_media': float(dm_apr.mean()),
                'falso_alarme_duas_piores': float(dd_apr.mean()),
            },
            'concordancia_media_x_duas_piores': {
                'videos_com_conteudo_toxico': concordancia(det_m, det_d),
                'videos_apropriados': concordancia(dm_apr, dd_apr),
                'spearman_scores_videos_toxicos': float(spearmanr(m, d2)[0]),
            },
            'concordancia_com_contagem_de_severas': {
                f'ao_menos_{c}': {'media': float((det_m == (n_sev >= c)).mean()),
                                  'duas_piores': float((det_d == (n_sev >= c)).mean())}
                for c in (1, 2, 3) if c <= k
            },
        }
    return {
        'variante': 'iid_nenhuma', 'f': F_TABELA,
        'distribuicao_scores_frase': distribuicao_scores(pop),
        'nenhuma_fracao_score_abaixo_0_5': float((nenhuma < 0.5).mean()) if nenhuma.size else None,
        'por_k': por_k,
        'leitura': (
            'Detecção pela média = soma em j de P(j Severas) * P(detecta | j). Compare, entre '
            'k=3 e k=5, o limiar calibrado (limiar_media), a mediana do score de um vídeo com '
            'exatamente uma Severa (video_com_exatamente_uma_severa.score_media_quantis.p50) e '
            'por_numero_de_severas.*.deteccao_media. Para (b): concordancia_media_x_duas_piores '
            'e concordancia_com_contagem_de_severas.'),
    }


def dispersao_video_seguro(pop):
    """Score e |ΔS| entre dois vídeos inteiramente apropriados (só Nenhuma), regra média."""
    out = {}
    for k in KS:
        rng = _rng(COD_DISPERSAO, COD_POP[pop['nome']], k)
        s = REGRAS['media'](videos_apropriados(rng, pop['scores'], pop['idx'][0], k, N_SIM))
        s2 = REGRAS['media'](videos_apropriados(rng, pop['scores'], pop['idx'][0], k, N_SIM))
        dif = np.abs(s - s2)
        out[str(k)] = {
            'media': float(s.mean()), 'desvio': float(s.std()),
            'p5': float(np.percentile(s, 5)),
            'diferenca_mediana_entre_videos': float(np.median(dif)),
            'quantis_diferenca_entre_videos': {f'p{q}': float(np.percentile(dif, q))
                                               for q in QUANTIS_DIF},
        }
    return out


# =============================================================================
# API usada pelo estudo de caso (analisar_estratos.py)
# =============================================================================

def limiar_e_banda(pop):
    """
    Limiar a priori e delta da banda a partir de uma população, com os mesmos
    geradores da simulação completa (mesmos números que o JSON grava).
    """
    lim = calibrar_celula(pop, 'nenhuma', K_SERVICO)['media']['limiar']
    disp = dispersao_video_seguro(pop)[str(K_SERVICO)]
    return lim, disp['quantis_diferenca_entre_videos'][f'p{PERCENTIL_BANDA}'], disp


def ler_simulacao_mais_recente(pasta):
    arquivos = sorted(Path(pasta).glob('simulacao_amostragem_*.json'),
                      key=lambda p: p.stat().st_mtime)
    if not arquivos:
        return None, None
    with open(arquivos[-1], encoding='utf-8') as fh:
        return json.load(fh), arquivos[-1]


# =============================================================================

def _imprimir_tabela(titulo, bloco):
    print(titulo)
    print(f'{"agregação":<16}' + ''.join(f'{f"k={k}":>22}' for k in KS))
    for regra in REGRAS:
        linha = f'{regra:<16}'
        for k in KS:
            d = bloco[regra][str(k)]
            ic = d.get('ic95_bootstrap_populacao') or [None, None]
            txt = f"{100 * d['deteccao']:.1f}"
            if ic[0] is not None:
                txt += f' [{100 * ic[0]:.1f},{100 * ic[1]:.1f}]'
            linha += f'{txt:>22}'
        print(linha)
    print()


def main():
    global N_SIM, N_BOOT_POP

    parser = argparse.ArgumentParser(description='Simulação de amostragem de frases por vídeo')
    parser.add_argument('--resultados', default=None,
                        help='Pasta da rodada com modelo_final/ (padrão: results/<tag>)')
    parser.add_argument('--modelo', default=None,
                        help='Modelo para o recurso de inferência no teste (padrão: o de '
                             'referência da rodada); só usado se não houver o CSV de teste')
    parser.add_argument('--saida', default=None,
                        help='Pasta de saída do JSON; tabelas em <saida>/latex/tabelas '
                             '(padrão: results/<tag> e experimento.dir_latex)')
    parser.add_argument('--idioma', choices=['pt', 'en'], default='en',
                        help='Idioma das tabelas LaTeX (padrão: en, o do manuscrito)')
    parser.add_argument('--n-sim', type=int, default=N_SIM,
                        help=f'Vídeos simulados por célula (padrão {N_SIM})')
    parser.add_argument('--n-boot', type=int, default=N_BOOT_POP,
                        help=f'Reamostragens bootstrap da população (padrão {N_BOOT_POP}; '
                             '0 desliga)')
    args = parser.parse_args()
    N_SIM, N_BOOT_POP = args.n_sim, args.n_boot

    t0 = time.time()
    pasta_res = Path(args.resultados) if args.resultados else experimento.dir_resultados()
    pasta_saida = Path(args.saida) if args.saida else experimento.dir_resultados()
    pasta_saida.mkdir(parents=True, exist_ok=True)

    pops, arquivos = carregar_populacoes(pasta_res, args.modelo)
    for nome, pop in pops.items():
        i = pop['info']
        print(f"População {nome}: {i['n']} frases ({i['n_por_classe']['Nenhuma']} Nenhuma, "
              f"{i['n_por_classe']['Leve']} Leve, {i['n_por_classe']['Severa']} Severa) "
              f"<- {Path(i['origem']).name}")
    if 'oof' not in pops:
        print('AVISO: sem predições OOF; limiar a priori e banda saem da população de teste.')
    print()

    try:
        desc_exp = experimento.descricao()
    except Exception:  # noqa: BLE001 -- variável de ambiente inválida não deve parar a simulação
        desc_exp = None

    resultado = {
        'timestamp': datetime.now().isoformat(),
        'script': 'app/nlp/evaluation/scripts/simular_amostragem.py',
        'experimento': desc_exp,
        'entradas': {'pasta_resultados': str(pasta_res),
                     'modelo_final_ts': (arquivos or {}).get('ts')},
        'semente': SEED, 'n_simulacoes': N_SIM, 'falso_alarme_alvo_pct': FALSO_ALARME_ALVO,
        'pesos_score': PESOS,
        'ks': list(KS), 'fracoes': list(FRACOES),
        'fracoes_chave': [f'{f:.2f}' for f in FRACOES], 'f_tabela': F_TABELA,
        'regras': DESCRICAO_REGRAS,
        'variantes': VARIANTES,
        'hipoteses': HIPOTESES,
        'populacoes': {n: p['info'] for n, p in pops.items()},
        'n_populacao': pops['teste']['info']['n'],
        'resultados': {},
        'bootstrap_populacao': {'n_reamostragens': N_BOOT_POP, 'n_simulacoes_por_celula': N_SIM_BOOT,
                                'estratificado_por_classe': True,
                                'intervalo': 'percentis 2,5 e 97,5'} if N_BOOT_POP else None,
        'diagnostico_m5b': {},
        'dispersao_video_seguro_por_populacao': {},
    }

    ic_extra = {}
    for nome, pop in pops.items():
        res = simular_populacao(pop)
        if N_BOOT_POP:
            ic_extra[nome] = bootstrap_populacao(pop, res)
        resultado['resultados'][nome] = res
        resultado['diagnostico_m5b'][nome] = diagnostico_m5b(pop, res['limiares'])
        resultado['dispersao_video_seguro_por_populacao'][nome] = dispersao_video_seguro(pop)

    # Chaves legadas (formato da versão submetida): população de teste, variante i.i.d.
    leg = resultado['resultados']['teste']['deteccao']['iid_nenhuma']
    resultado['deteccao'] = {fk: {regra: {k: {'limiar': d['limiar'], 'deteccao': d['deteccao']}
                                          for k, d in por_k.items()}
                                  for regra, por_k in bloco.items()}
                             for fk, bloco in leg.items()}
    resultado['dispersao_video_seguro'] = resultado['dispersao_video_seguro_por_populacao']['teste']

    # ---- limiar a priori (M4b) e banda de tolerância (M7c) ----
    pop_ref = 'oof' if 'oof' in pops else 'teste'
    lim_ap = resultado['resultados'][pop_ref]['limiares']['nenhuma']['media'][str(K_SERVICO)]
    disp_ref = resultado['dispersao_video_seguro_por_populacao'][pop_ref][str(K_SERVICO)]
    aviso = None if pop_ref == 'oof' else (
        'Sem predições OOF nesta rodada: o valor saiu da população de teste. Não é '
        'reuso do estudo de caso, mas não é o protocolo pretendido (OOF).')
    resultado['limiar_a_priori'] = {
        'valor': lim_ap['limiar'],
        'definicao': (f'percentil {FALSO_ALARME_ALVO} do score do vídeo (média das k={K_SERVICO} '
                      f'frases) em {N_SIM} vídeos sintéticos inteiramente apropriados (só '
                      f'frases Nenhuma), população {pop_ref}; regra: score < limiar => rebaixado'),
        'populacao': pop_ref, 'k': K_SERVICO, 'regra': 'media', 'base_apropriada': 'nenhuma',
        'falso_alarme_alvo_pct': FALSO_ALARME_ALVO,
        'falso_alarme_verificado': lim_ap['falso_alarme_verificado'],
        'ic95_bootstrap_populacao': (ic_extra.get(pop_ref) or {}).get('limiar_media_k3_nenhuma'),
        'alternativas': {
            f'{p}_{b}': resultado['resultados'][p]['limiares'][b]['media'][str(K_SERVICO)]['limiar']
            for p in resultado['resultados'] for b in COD_BASE},
        'aviso': aviso,
    }
    resultado['banda_tolerancia'] = {
        'delta': disp_ref['quantis_diferenca_entre_videos'][f'p{PERCENTIL_BANDA}'],
        'definicao': (f'percentil {PERCENTIL_BANDA} de |S1 - S2| entre dois vídeos sintéticos '
                      f'inteiramente apropriados (só Nenhuma) e independentes, k={K_SERVICO}, '
                      f'regra média, população {pop_ref}'),
        'regra_de_reordenacao': (
            'u fica acima de v se S(u) - S(v) > delta; entre os demais pares vale a ordem da '
            'plataforma. Ordem final: extensão linear lexicograficamente mínima (pela posição na '
            'plataforma) dessa ordem parcial -- a cada passo entra, entre os vídeos restantes '
            'que nenhum outro restante supera por mais de delta, o de melhor posição na '
            'plataforma (analisar_reordenacao.ordenar_com_banda).'),
        'populacao': pop_ref, 'k': K_SERVICO, 'percentil': PERCENTIL_BANDA,
        'quantis_diferenca': disp_ref['quantis_diferenca_entre_videos'],
        'ic95_bootstrap_populacao': (ic_extra.get(pop_ref) or {}).get('banda_p95_k3'),
        'aviso': aviso,
    }
    resultado['segundos'] = round(time.time() - t0, 1)

    # ---- saída no console ----
    for nome in resultado['resultados']:
        _imprimir_tabela(f'--- população {nome}: f = {F_TABELA:.2f}, i.i.d., apropriadas só '
                         f'Nenhuma (detecção %, IC bootstrap da população) ---',
                         resultado['resultados'][nome]['deteccao']['iid_nenhuma'][f'{F_TABELA:.2f}'])
    nome_var = pop_ref
    print(f'--- variantes, regra média, população {nome_var} (detecção %) ---')
    print(f'{"variante":<22}{"f":>6}' + ''.join(f'{f"k={k}":>9}' for k in KS))
    for variante in VARIANTES:
        for f in FRACOES:
            bloco = resultado['resultados'][nome_var]['deteccao'][variante][f'{f:.2f}']['media']
            print(f'{variante:<22}{f:>6.2f}' +
                  ''.join(f"{100 * bloco[str(k)]['deteccao']:>9.1f}" for k in KS))
    print()
    for nome, diag in resultado['diagnostico_m5b'].items():
        print(f'--- diagnóstico M5b, população {nome} (f = {F_TABELA:.2f}) ---')
        for k, d in diag['por_k'].items():
            c = d['concordancia_media_x_duas_piores']['videos_com_conteudo_toxico']['concordancia']
            u = d['video_com_exatamente_uma_severa']
            print(f"  k={k:<3} limiar média {d['limiar_media']:.3f} | 1 Severa: mediana "
                  f"{u['score_media_quantis']['p50']:.3f}, detectado {100 * (u['fracao_abaixo_do_limiar_media'] or 0):.1f}% "
                  f"| P(>=1 Sev) {100 * d['p_ao_menos_uma_severa']:.1f}%  P(>=2) "
                  f"{100 * d['p_ao_menos_duas_severas']:.1f}% | concordância média x 2 piores "
                  f"{100 * c:.1f}%")
    print()
    la, bt = resultado['limiar_a_priori'], resultado['banda_tolerancia']
    print(f"Limiar a priori ({la['populacao']}, k={K_SERVICO}, média, {FALSO_ALARME_ALVO}% FA): "
          f"{la['valor']:.4f}  IC95 {la['ic95_bootstrap_populacao']}")
    print(f"Banda de tolerância delta ({bt['populacao']}, p{PERCENTIL_BANDA} |ΔS|, k={K_SERVICO}): "
          f"{bt['delta']:.4f}  IC95 {bt['ic95_bootstrap_populacao']}")
    print(f"Tempo: {resultado['segundos']:.0f} s")

    destino = pasta_saida / f'simulacao_amostragem_{datetime.now():%Y%m%d_%H%M%S}.json'
    with open(destino, 'w', encoding='utf-8') as fh:
        json.dump(resultado, fh, indent=2, ensure_ascii=False)
    print(f'\nSalvo em: {destino}')

    saidas = importar_irmao('regerar_tabelas_estudo_caso')
    for caminho in saidas.escrever_saidas_simulacao(resultado, args.idioma, args.saida,
                                                    f'{F_TABELA:.2f}'):
        print(f'Tabela LaTeX: {caminho}')


if __name__ == '__main__':
    main()
