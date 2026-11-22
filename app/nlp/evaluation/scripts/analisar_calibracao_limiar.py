"""
Limiar da classe Severa escolhido fora do teste, curva precisão-revocação,
calibração do detector e sensibilidade do score aos pesos (revisão do JBCS).

Atende aos itens M6e, m6, m7, M7a e M7b do plano de resposta aos revisores
(documentos/plano-resposta-revisores.md, pacotes P4 e P8). Não carrega modelo nem
usa GPU: trabalha só sobre as probabilidades gravadas pelo treino final
(train_toxicity_gridsearch.py --mode final), em results/<tag>/modelo_final/:

  probabilidades_oof_<ts>.csv    predições out-of-fold sobre a partição de treino
                                 (folds agrupados por vídeo, semente de referência)
  probabilidades_teste_<ts>.csv  probabilidades no teste retido, uma trinca de
                                 colunas por semente (p_nenhuma_s42, ...)
  modelo_final_<ts>.json         configuração, sementes e semente de referência

O QUE CALCULA
-------------
  1. (M6e) Limiar tau sobre P(Severa) escolhido nas predições OOF, por duas regras
     declaradas antes de ver as OOF do corpus v1 (constantes abaixo):
       (a) o MENOR tau da grade cujo falso alarme OOF (frase Nenhuma sinalizada
           como Severa) fique <= falso alarme OOF do argmax + ORCAMENTO_FA_PP
           pontos percentuais -- é a regra principal;
       (b) o tau que maximiza o F2 da classe Severa nas OOF (empate: menor tau).
     Para cada tau escolhido, reporta no teste retido: recall da Severa, Severas
     não sinalizadas, falso alarme sob a regra "Severa" e sob a regra "tóxico em
     qualquer grau". Os limiares escolhidos no próprio teste aparecem só como
     comparação exploratória.
  2. (m7) Curva de limiar numa grade fina (0,01 a 0,99, passo 0,01), no teste e nas
     OOF; curva precisão-revocação da classe Severa com precisão média (AP).
  3. (m6) Frases Severa do teste que o modelo de referência prediz como Nenhuma,
     com texto (do corpus, via id_sentenca), p_severa e score.
  4. (M7a) ECE (15 bins de largura igual; top-label e por classe, em especial a
     Severa), Brier multiclasse, log-verossimilhança negativa e diagrama de
     confiabilidade, nas OOF e no teste. Temperature scaling ajustado nas OOF
     (logits recuperados como log-probabilidades: softmax(log p / T) é o
     temperature scaling a menos de uma constante por frase) e aplicado ao teste;
     efeito no score P = 1 - (0,5*p_leve + 1,0*p_severa).
  5. (M7b, nível da frase) Sensibilidade do score ao peso da classe Leve
     (w1 em {0,25; 0,5; 0,75}, Nenhuma 0 e Severa 1): AUC, mediana por classe e
     correlação de Spearman entre os ordenamentos.

SAÍDAS
------
  results/<tag>/calibracao_limiar_<ts>.json      todos os números, chaves explícitas
  dir_latex('figuras')/fig-pr-severa.png         curva precisão-revocação (Severa)
  dir_latex('figuras')/fig-limiar-severa.png     recall e falso alarme contra tau
  dir_latex('figuras')/fig-calibracao.png        diagrama de confiabilidade
  dir_latex('tabelas')/tab-limiar.tex            Tabela 6 (com as OOF)
  dir_latex('tabelas')/tab-pontos-operacao.tex   Tabela 7 (tau escolhido nas OOF)

  Com --idioma pt as três figuras saem com rótulos em português e sufixo -pt; a opção
  exige --saida, para não gravar um JSON novo nem reescrever tabelas na pasta da rodada.

As duas tabelas do manuscrito passaram a ser deste script. O
analisar_pontos_operacao.py, que escolhia tau no teste, continua no pipeline mas
grava as versões dele com o sufixo -teste (exploratórias), sem sobrescrever estas.

Uso (de src/). Rode pelo CAMINHO do arquivo: com `python -m`, o __init__ do pacote
app.nlp.evaluation importa torch e transformers (ModelEvaluator) sem necessidade --
com a memória tomada por um treino em GPU, isso chegou a falhar ao carregar as DLLs.
    python app/nlp/evaluation/scripts/analisar_calibracao_limiar.py
    # validação com as entradas de outra rodada (ids não batem com o corpus v1;
    # o texto das frases fica de fora e a saída vai para outra pasta):
    python app/nlp/evaluation/scripts/analisar_calibracao_limiar.py \
        --resultados app/nlp/evaluation/results/video-original --saida <pasta>
(`python -m app.nlp.evaluation.scripts.analisar_calibracao_limiar` também funciona,
só que importa torch.)
"""

import os

# Análise sobre probabilidades gravadas: nunca precisa de GPU. Rodado pelo caminho
# do arquivo, nada aqui importa torch. Com `python -m`, o __init__ do pacote
# app.nlp.evaluation importa torch ANTES deste módulo (sem inicializar CUDA, que é
# preguiçosa); esta linha garante que nada daqui a inicialize.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import (average_precision_score, precision_recall_curve,  # noqa: E402
                             roc_auc_score)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp import experimento  # noqa: E402

CLASSES = ['Nenhuma', 'Leve', 'Severa']
COLS = ['p_nenhuma', 'p_leve', 'p_severa']
PESOS = (0.0, 0.5, 1.0)

# --- Regras declaradas a priori (antes de existirem as OOF do corpus v1) -------
# Grade de limiares: 0,01 a 0,99, passo 0,01. Sinaliza-se Severa se P(Severa) > tau.
GRADE_TAU = np.round(np.arange(1, 100) / 100.0, 2)
# Regra (a): folga de falso alarme (Nenhuma sinalizada como Severa) sobre o argmax,
# em pontos percentuais. Fixada em 2 p.p. antes de ver as OOF do v1.
ORCAMENTO_FA_PP = 2.0
# Regra (b): beta do F-beta da classe Severa (beta = 2: recall pesa mais, porque
# deixar passar conteúdo severo é o erro caro).
BETA_F = 2.0
REGRA_PRINCIPAL = 'a_orcamento_fa'

N_BINS_ECE = 15
PESOS_LEVE_SENSIBILIDADE = (0.25, 0.5, 0.75)
# Pontos fixos da grade candidatos a linha da Tabela 6 (ilustram o compromisso no
# teste; nenhum é escolhido). Linhas com desfecho idêntico no teste são omitidas.
TAUS_TABELA = (0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01)
# Linha de grade que sinaliza mais que esta fração das Nenhuma no teste é
# degenerada (o piso de P(Severa) sob label smoothing fica perto de 0,05) e não
# entra na tabela; a curva inteira fica no JSON e na figura.
FA_MAX_LINHA_GRADE = 0.50
# No diagrama de confiabilidade, bins com menos frases que isto não são desenhados
# (um bin de 1 a 3 frases salta entre 0 e 1 e domina a leitura); o ECE usa todos.
MIN_N_BIN_FIGURA = 5

# Paleta e tinta do sistema de figuras do artigo (ver gerar_figuras_artigo.py).
COR_TESTE = '#2a78d6'
COR_OOF = '#eb6834'
COR_CALIBRADO = '#1baf7a'
INK = '#0b0b0b'
INK_SUAVE = '#52514e'
GRADE = '#e1e0d9'
EIXO = '#c3c2b7'

# Idioma dos rótulos das figuras (--idioma). O padrão é o inglês do manuscrito; com
# 'pt' as mesmas figuras saem com rótulos em português, vírgula decimal e sufixo -pt.
# Os números do JSON e as tabelas não mudam com o idioma.
IDIOMA = 'en'


def _t(en: str, pt: str) -> str:
    """Texto de rótulo no idioma das figuras."""
    return en if IDIOMA == 'en' else pt


def _n(valor: float, casas: int) -> str:
    """Número com o separador decimal do idioma das figuras."""
    texto = f'{valor:.{casas}f}'
    return texto if IDIOMA == 'en' else texto.replace('.', ',')


def _g(valor: float) -> str:
    """Como f'{valor:g}', com o separador decimal do idioma das figuras."""
    texto = f'{valor:g}'
    return texto if IDIOMA == 'en' else texto.replace('.', ',')


def _marcas_unitarias(ax):
    """Em português, marcas de 0 a 1 com vírgula; em inglês o eixo fica no padrão."""
    if IDIOMA == 'en':
        return
    marcas = np.round(np.arange(0, 1.01, 0.2), 1)
    rotulos = [_n(m, 1) for m in marcas]
    ax.set_xticks(marcas)
    ax.set_xticklabels(rotulos)
    ax.set_yticks(marcas)
    ax.set_yticklabels(rotulos)


# =============================================================================
# Entradas
# =============================================================================

def localizar_entradas(pasta: Path, ts: str = None):
    """Arquivos do treino final com o mesmo timestamp (o mais recente, por padrão)."""
    pasta_mf = pasta / 'modelo_final' if (pasta / 'modelo_final').is_dir() else pasta
    if ts is None:
        jsons = sorted(pasta_mf.glob('modelo_final_*.json'))
        if jsons:
            ts = jsons[-1].stem[len('modelo_final_'):]
        else:
            oofs = sorted(pasta_mf.glob('probabilidades_oof_*.csv'))
            if not oofs:
                raise FileNotFoundError(f"Nenhuma probabilidades_oof_*.csv em {pasta_mf}. "
                                        "Rode antes o treino final (--mode final).")
            ts = oofs[-1].stem[len('probabilidades_oof_'):]
    arq_oof = pasta_mf / f'probabilidades_oof_{ts}.csv'
    arq_teste = pasta_mf / f'probabilidades_teste_{ts}.csv'
    arq_json = pasta_mf / f'modelo_final_{ts}.json'
    for a in (arq_oof, arq_teste):
        if not a.exists():
            raise FileNotFoundError(f"Arquivo esperado não encontrado: {a}")
    return ts, arq_oof, arq_teste, (arq_json if arq_json.exists() else None)


def _normalizar(P):
    # As probabilidades vêm do CSV com ~8 algarismos (float32); renormaliza cada
    # linha para somar 1 exatamente antes de tomar log ou calcular Brier.
    P = np.asarray(P, dtype=float)
    return P / P.sum(axis=1, keepdims=True)


def carregar_probabilidades(arq_oof, arq_teste, semente_ref=None):
    oof = pd.read_csv(arq_oof, sep=';', dtype={'video_id': str})
    validas = oof[COLS].notna().all(axis=1)
    descartadas = int((~validas).sum())
    oof = oof[validas].reset_index(drop=True)

    te = pd.read_csv(arq_teste, sep=';')
    sementes = [int(m.group(1)) for c in te.columns
                for m in [re.fullmatch(r'p_nenhuma_s(\d+)', c)] if m]
    if not sementes:
        raise ValueError(f"Nenhuma coluna p_nenhuma_s<semente> em {arq_teste}")
    if semente_ref is None:
        semente_ref = sementes[0]
    if semente_ref not in sementes:
        raise ValueError(f"Semente de referência {semente_ref} ausente do CSV de teste "
                         f"(sementes: {sementes})")
    P_te = {}
    for s in sementes:
        bloco = te[[f'{c}_s{s}' for c in COLS]]
        if bloco.isna().any().any():
            raise ValueError(f"Probabilidades ausentes no teste para a semente {s}")
        P_te[s] = _normalizar(bloco.to_numpy())
    return {
        'ids_oof': oof['id_sentenca'].to_numpy(int),
        'y_oof': oof['rotulo'].to_numpy(int),
        'P_oof': _normalizar(oof[COLS].to_numpy()),
        'oof_descartadas_nan': descartadas,
        'ids_teste': te['id_sentenca'].to_numpy(int),
        'y_teste': te['rotulo'].to_numpy(int),
        'P_teste': P_te,
        'sementes': sementes,
        'semente_ref': semente_ref,
    }


def carregar_llm(pasta: Path, n_teste: int):
    """Matriz de confusão do modelo generativo (llm_baselines_toxicity_*.json mais recente)."""
    fs = sorted(pasta.glob('llm_baselines_toxicity_*.json'))
    if not fs:
        return None
    with open(fs[-1], encoding='utf-8') as f:
        d = json.load(f)
    nome, r = next(iter(d['results'].items()))
    m = np.asarray(r['confusion_matrix'])
    n = int(r.get('n_amostras') or m.sum())
    return {
        'nome': nome, 'confusion_matrix': m.tolist(), 'n_amostras': n,
        'segundos_por_frase': round(r['segundos'] / n, 2) if r.get('segundos') else None,
        'fonte': fs[-1].name,
        'mesmo_n_do_teste': bool(int(m.sum()) == n_teste),
    }


def carregar_textos(ids, y):
    """Texto das frases do teste pelo id_sentenca, conferindo o rótulo linha a linha."""
    from app.nlp.datasets.prepare_data_toxicity import load_toxicity_corpus

    df = load_toxicity_corpus().set_index('id_sentenca')
    faltando = sorted(set(int(i) for i in ids) - set(df.index))
    if faltando:
        return None, f"{len(faltando)} id_sentenca do CSV ausentes do corpus (ex.: {faltando[:5]})"
    rotulos = df.loc[ids, 'label'].to_numpy(int)
    divergentes = int((rotulos != np.asarray(y)).sum())
    if divergentes:
        return None, (f"{divergentes} frases com rótulo do CSV diferente do corpus: "
                      "as probabilidades não são deste corpus")
    return {int(i): str(t) for i, t in zip(ids, df.loc[ids, 'texto'])}, None


# =============================================================================
# Métricas
# =============================================================================

def _div(a, b):
    return float(a) / float(b) if b else None


def wilson(k, n, z=1.959963984540054):
    """IC de 95% de Wilson para uma proporção k/n."""
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    meia = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, centro - meia), min(1.0, centro + meia)]


def taxas_limiar(y, P, tau):
    """Sinaliza Severa quando P(Severa) > tau (estrito)."""
    pred = P.argmax(axis=1)
    sinal = P[:, 2] > tau
    sev, nen = y == 2, y == 0
    n_sev, n_nen = int(sev.sum()), int(nen.sum())
    tp = int((sinal & sev).sum())
    fp = int((sinal & ~sev).sum())
    fp_nen = int((sinal & nen).sum())
    tox_nen = int(((pred != 0) | sinal)[nen].sum())
    precisao = _div(tp, tp + fp)
    recall = _div(tp, n_sev)
    b2 = BETA_F ** 2
    fbeta = (((1 + b2) * precisao * recall / (b2 * precisao + recall))
             if precisao is not None and recall is not None and (precisao + recall) > 0
             else None)
    fpr_nao_sev = _div(fp, int((~sev).sum()))
    return {
        'tau': float(tau),
        'recall_severa': recall,
        'severas_nao_sinalizadas': n_sev - tp,
        # Severas que passariam como apropriadas: nem Severa pelo limiar, nem
        # tóxicas pelo argmax.
        'severas_como_nenhuma': int((sev & ~sinal & (pred == 0)).sum()),
        'fa_severa': _div(fp_nen, n_nen),
        'fa_toxico': _div(tox_nen, n_nen),
        'precisao_severa': precisao,
        'f2_severa': fbeta,
        'youden_severa': (recall - fpr_nao_sev) if recall is not None and fpr_nao_sev is not None else None,
        'n_severa': n_sev, 'n_nenhuma': n_nen,
        'n_severas_sinalizadas': tp, 'n_nenhuma_sinalizadas_severa': fp_nen,
        'n_nenhuma_sinalizadas_toxico': tox_nen,
    }


def taxas_argmax(y, P):
    pred = P.argmax(axis=1)
    sev, nen = y == 2, y == 0
    n_sev, n_nen = int(sev.sum()), int(nen.sum())
    tp = int((pred[sev] == 2).sum())
    fp_nen = int((pred[nen] == 2).sum())
    tox_nen = int((pred[nen] != 0).sum())
    return {
        'regra': 'argmax',
        'recall_severa': _div(tp, n_sev),
        'severas_nao_sinalizadas': n_sev - tp,
        'severas_como_nenhuma': int((pred[sev] == 0).sum()),
        'fa_severa': _div(fp_nen, n_nen),
        'fa_toxico': _div(tox_nen, n_nen),
        'precisao_severa': _div(tp, int((pred == 2).sum())),
        'n_severa': n_sev, 'n_nenhuma': n_nen,
        'n_severas_sinalizadas': tp, 'n_nenhuma_sinalizadas_severa': fp_nen,
        'n_nenhuma_sinalizadas_toxico': tox_nen,
    }


def com_ic(t):
    """Acrescenta IC de Wilson às proporções de um ponto de operação."""
    t = dict(t)
    t['ic95_wilson'] = {
        'recall_severa': wilson(t['n_severas_sinalizadas'], t['n_severa']),
        'fa_severa': wilson(t['n_nenhuma_sinalizadas_severa'], t['n_nenhuma']),
        'fa_toxico': wilson(t['n_nenhuma_sinalizadas_toxico'], t['n_nenhuma']),
    }
    return t


def curva_limiar(y, P):
    return [taxas_limiar(y, P, t) for t in GRADE_TAU]


def escolher_tau_orcamento(curva, fa_argmax):
    """Regra (a): menor tau com falso alarme (Severa) <= argmax + orçamento."""
    teto = fa_argmax + ORCAMENTO_FA_PP / 100.0
    viaveis = [l for l in curva if l['fa_severa'] is not None and l['fa_severa'] <= teto + 1e-12]
    # Sempre há solução: P(Severa) > 0,5 implica argmax = Severa, então o falso
    # alarme em tau >= 0,5 nunca passa do argmax.
    return min(viaveis, key=lambda l: l['tau'])['tau'], teto


def escolher_tau_fbeta(curva):
    """Regra (b): tau que maximiza o F2 da Severa; empate -> menor tau."""
    validos = [l for l in curva if l['f2_severa'] is not None]
    melhor = max(l['f2_severa'] for l in validos)
    empatados = [l for l in validos if l['f2_severa'] >= melhor - 1e-12]
    return min(empatados, key=lambda l: l['tau'])['tau']


def resumo_sementes(lista, chaves=('recall_severa', 'severas_nao_sinalizadas',
                                    'fa_severa', 'fa_toxico')):
    out = {}
    for k in chaves:
        v = np.array([d[k] for d in lista if d.get(k) is not None], dtype=float)
        if len(v):
            out[k] = {'media': float(v.mean()),
                      'dp': float(v.std(ddof=1)) if len(v) > 1 else None,
                      'min': float(v.min()), 'max': float(v.max())}
    return out


def curva_pr(y, P):
    alvo = (y == 2).astype(int)
    prec, rec, thr = precision_recall_curve(alvo, P[:, 2])
    return {
        'average_precision': float(average_precision_score(alvo, P[:, 2])),
        'prevalencia_severa': float(alvo.mean()),
        'nota': ("pontos de sklearn.metrics.precision_recall_curve (sinaliza com "
                 "p_severa >= limiar); 'limiares' tem um elemento a menos que "
                 "'precisao' e 'recall', cujo último ponto é (recall 0, precisão 1)"),
        'precisao': [round(float(v), 6) for v in prec],
        'recall': [round(float(v), 6) for v in rec],
        'limiares': [round(float(v), 6) for v in thr],
    }


# --- calibração -------------------------------------------------------------

def _ece(conf, alvo, n_bins=N_BINS_ECE):
    """ECE com bins de largura igual: [0, 1/15], (1/15, 2/15], ..., (14/15, 1]."""
    conf = np.asarray(conf, dtype=float)
    alvo = np.asarray(alvo, dtype=float)
    bordas = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(conf, bordas[1:-1], right=True)
    ece, bins = 0.0, []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n:
            c, f = float(conf[m].mean()), float(alvo[m].mean())
            ece += n / len(conf) * abs(f - c)
        else:
            c = f = None
        bins.append({'lim_inf': float(bordas[b]), 'lim_sup': float(bordas[b + 1]),
                     'n': n, 'confianca_media': c, 'frequencia_observada': f})
    return float(ece), bins


def brier(y, P):
    um = np.eye(P.shape[1])[y]
    return float(np.mean(np.sum((P - um) ** 2, axis=1)))


def nll(y, P):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0))))


def metricas_calibracao(y, P, com_bins=True):
    ece_top, bins_top = _ece(P.max(axis=1), P.argmax(axis=1) == y)
    por_classe, bins_classe = {}, {}
    for k, nome in enumerate(CLASSES):
        e, b = _ece(P[:, k], y == k)
        por_classe[nome] = e
        bins_classe[nome] = b
    out = {
        'ece_top_label': ece_top,
        'ece_severa': por_classe['Severa'],
        'ece_por_classe': por_classe,
        'ece_por_classe_media': float(np.mean(list(por_classe.values()))),
        'brier_multiclasse': brier(y, P),
        'nll': nll(y, P),
        'acuracia': float((P.argmax(axis=1) == y).mean()),
        'confianca_media_top_label': float(P.max(axis=1).mean()),
    }
    if com_bins:
        out['bins_top_label'] = bins_top
        out['bins_severa'] = bins_classe['Severa']
    return out


def aplicar_temperatura(P, T):
    L = np.log(np.clip(P, 1e-12, 1.0)) / T
    L -= L.max(axis=1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(axis=1, keepdims=True)


def ajustar_temperatura(y, P):
    """
    Temperatura que minimiza a NLL nas OOF. Otimiza em beta = 1/T, no qual a NLL é
    convexa (log-soma-exp de uma função linear em beta), em [0,05; 20].
    """
    lim = (0.05, 20.0)
    res = minimize_scalar(lambda b: nll(y, aplicar_temperatura(P, 1.0 / b)),
                          bounds=lim, method='bounded', options={'xatol': 1e-7})
    beta = float(res.x)
    return {
        'T': 1.0 / beta,
        'metodo': ("NLL mínima nas OOF, logits = log p (softmax(log p / T) equivale ao "
                   "temperature scaling dos logits originais a menos de constante por frase); "
                   "otimização limitada em beta = 1/T"),
        'limites_beta': list(lim),
        'convergiu': bool(res.success),
        'no_limite': bool(min(abs(beta - lim[0]), abs(beta - lim[1])) < 1e-4),
        'nll_oof_antes': nll(y, P),
        'nll_oof_depois': float(res.fun),
    }


# --- score ------------------------------------------------------------------

def score(P, w_leve=0.5, w_severa=1.0):
    return 1.0 - (w_leve * P[:, 1] + w_severa * P[:, 2])


def _auc(alvo, s):
    alvo = np.asarray(alvo).astype(int)
    if alvo.min() == alvo.max():
        return None
    return float(roc_auc_score(alvo, s))


def resumo_score(y, s):
    """AUC com positivo = classe menos tóxica (score maior = mais seguro)."""
    por_classe = {}
    for k, nome in enumerate(CLASSES):
        v = s[y == k]
        por_classe[nome] = ({'n': int(len(v)), 'mediana': float(np.median(v)),
                             'q1': float(np.percentile(v, 25)),
                             'q3': float(np.percentile(v, 75))} if len(v) else {'n': 0})
    m_ns = (y == 0) | (y == 2)
    m_ls = (y == 1) | (y == 2)
    return {
        'por_classe': por_classe,
        'auc_nenhuma_vs_toxica': _auc(y == 0, s),
        'auc_nenhuma_vs_severa': _auc(y[m_ns] == 0, s[m_ns]),
        'auc_leve_vs_severa': _auc(y[m_ls] == 1, s[m_ls]),
    }


def _spearman(a, b):
    return float(spearmanr(a, b).correlation)


def sensibilidade_pesos(y, P):
    scores = {w: score(P, w_leve=w) for w in PESOS_LEVE_SENSIBILIDADE}
    out = {'por_peso_leve': {f'{w:.2f}': resumo_score(y, s) for w, s in scores.items()},
           'spearman_entre_pesos': {}}
    ws = list(PESOS_LEVE_SENSIBILIDADE)
    for i in range(len(ws)):
        for j in range(i + 1, len(ws)):
            out['spearman_entre_pesos'][f'{ws[i]:.2f}_vs_{ws[j]:.2f}'] = \
                _spearman(scores[ws[i]], scores[ws[j]])
    return out


def efeito_temperatura_no_score(y, P, P_cal):
    s0, s1 = score(P), score(P_cal)
    d = np.abs(s1 - s0)
    return {
        'antes': resumo_score(y, s0),
        'depois': resumo_score(y, s1),
        'spearman_antes_depois': _spearman(s0, s1),
        'delta_abs_medio': float(d.mean()),
        'delta_abs_mediano': float(np.median(d)),
        'delta_abs_max': float(d.max()),
        'argmax_preservado': bool((P.argmax(axis=1) == P_cal.argmax(axis=1)).all()),
    }


# --- frases Severa preditas como Nenhuma (m6) -------------------------------

def severas_preditas_nenhuma(ids, y, P, textos, tau_oof):
    pred = P.argmax(axis=1)
    s = score(P)
    idx = np.where((y == 2) & (pred == 0))[0]
    frases = [{
        'id_sentenca': int(ids[i]),
        'texto': textos.get(int(ids[i])) if textos else None,
        'p_nenhuma': float(P[i, 0]), 'p_leve': float(P[i, 1]), 'p_severa': float(P[i, 2]),
        'score': float(s[i]),
        'sinalizada_pelo_tau_oof': bool(P[i, 2] > tau_oof),
    } for i in idx]
    frases.sort(key=lambda d: d['score'])
    nenhuma_ok = s[(y == 0) & (pred == 0)]
    return {
        'n': len(frases),
        'n_severa_teste': int((y == 2).sum()),
        'frases': frases,
        'score_mediano_perdidas': float(np.median([f['score'] for f in frases])) if frases else None,
        'score_mediano_nenhuma_corretas': float(np.median(nenhuma_ok)) if len(nenhuma_ok) else None,
        'tau_oof_usado': float(tau_oof),
    }


# =============================================================================
# Tabelas LaTeX
# =============================================================================

def _pct(x):
    return f"{100 * x:.1f}\\%"


def _tex_escape(s):
    for a, b in (('\\', '\\textbackslash{}'), ('&', '\\&'), ('%', '\\%'),
                 ('#', '\\#'), ('_', '\\_')):
        s = s.replace(a, b)
    return s


ROTULO_ESCOLHA = {
    'argmax': '\\emph{argmax}',
    'a': 'OOF: false-alarm budget',
    'b': 'OOF: max.\\ $F_2$',
    'ab': 'OOF: both rules',
    'grade': 'fixed grid',
}


def linhas_tabela_limiar(y_oof, P_oof, y_te, P_te, tau_a, tau_b):
    def chave(t):
        return (t['n_severas_sinalizadas'], t['n_nenhuma_sinalizadas_severa'])

    linhas = [{'escolha': 'argmax', 'tau': None,
               'oof': taxas_argmax(y_oof, P_oof), 'teste': taxas_argmax(y_te, P_te)}]
    especiais = [('ab', tau_a)] if tau_a == tau_b else [('a', tau_a), ('b', tau_b)]
    for esc, t in especiais:
        linhas.append({'escolha': esc, 'tau': t, 'oof': taxas_limiar(y_oof, P_oof, t),
                       'teste': taxas_limiar(y_te, P_te, t)})
    vistos = {chave(l['teste']) for l in linhas}
    for t in TAUS_TABELA:
        if t in (tau_a, tau_b):
            continue
        te = taxas_limiar(y_te, P_te, t)
        if chave(te) in vistos or te['fa_severa'] > FA_MAX_LINHA_GRADE:
            continue
        vistos.add(chave(te))
        linhas.append({'escolha': 'grade', 'tau': t, 'oof': taxas_limiar(y_oof, P_oof, t),
                       'teste': te})
    return [linhas[0]] + sorted(linhas[1:], key=lambda l: -l['tau'])


def tex_limiar(linhas, argmax_coincide_050):
    ident = (" On this test set \\emph{argmax} flags the same sentences as $\\tau = 0.5$."
             if argmax_coincide_050 else "")
    cap = ("Effect of the threshold $\\tau$ on the probability of the Severe class, on the "
           "out-of-fold (OOF) predictions of the training split and on the held-out test set. "
           "Every threshold row flags by the same rule, $P(\\text{Severe}) > \\tau$, and false "
           "alarm is the fraction of None sentences so flagged. The rows marked OOF were chosen "
           "on the out-of-fold predictions alone, by rules fixed in advance: the smallest "
           f"$\\tau$ whose OOF false alarm stays within {ORCAMENTO_FA_PP:g} percentage points "
           "of that of \\emph{argmax} (false-alarm budget), and the $\\tau$ that maximises the "
           "OOF $F_2$ of the Severe class. Fixed-grid rows illustrate the trade-off on the "
           "test set and were not used to choose anything; grid rows whose test outcome "
           "repeats another row, or that flag more than half of the None sentences, are "
           "omitted." + ident)
    out = ["% Gerado por app/nlp/evaluation/scripts/analisar_calibracao_limiar.py",
           "% NAO EDITAR A MAO: reexecute o script apos retreinar.",
           "\\begin{table*}[!t]", "\\centering", f"\\caption{{{cap}}}",
           "\\label{tab:limiar}", "\\begin{tabular}{llccccc}", "\\toprule",
           " & & \\multicolumn{2}{c}{Out-of-fold (training split)} & "
           "\\multicolumn{3}{c}{Held-out test} \\\\",
           "\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}",
           "$\\tau$ & Chosen by & Severe recall & False alarm & Severe recall & "
           "Severe not flagged & False alarm \\\\",
           "\\midrule"]
    for l in linhas:
        o, t = l['oof'], l['teste']
        cells = ["---" if l['tau'] is None else f"{l['tau']:.2f}",
                 ROTULO_ESCOLHA[l['escolha']],
                 _pct(o['recall_severa']), _pct(o['fa_severa']),
                 _pct(t['recall_severa']), str(t['severas_nao_sinalizadas']),
                 _pct(t['fa_severa'])]
        if l['escolha'] in ('a', 'ab'):
            cells = [f"\\textbf{{{c}}}" for c in cells]
        out.append(" & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(out)


def tex_pontos(pontos):
    cap = ("Operating points on the held-out test set. The expensive error is letting severely "
           "toxic content through; the cheap one is flagging appropriate content, which merely "
           "demotes it in the ranking. The thresholds were chosen on out-of-fold predictions of "
           "the training split, before the test set was consulted: the false-alarm-budget rule "
           f"takes the smallest $\\tau$ whose out-of-fold false alarm stays within "
           f"{ORCAMENTO_FA_PP:g} percentage points of that of \\emph{{argmax}}, and the $F_2$ rule "
           "maximises the out-of-fold $F_2$ of the Severe class. False alarm is reported under "
           "both flagging rules, because the rows do not flag alike: \\emph{argmax} and the "
           "generative model return a class, whereas the threshold is applied to "
           "$P(\\text{Severe})$; under the any-level rule a thresholded row also counts the "
           "sentences that \\emph{argmax} assigns to Mild.")
    cab = ("Model & Severe recall & Severe missed & \\multicolumn{2}{c}{Appropriate flagged} \\\\\n"
           "\\cmidrule(lr){4-5}\n"
           " & & & as Severe & as toxic (any) ")
    out = ["% Gerado por app/nlp/evaluation/scripts/analisar_calibracao_limiar.py",
           "% NAO EDITAR A MAO: reexecute o script apos retreinar.",
           "\\begin{table*}[!t]", "\\centering", f"\\caption{{{cap}}}",
           "\\label{tab:pontos-operacao}", "\\begin{tabular}{lcccc}", "\\toprule",
           cab + " \\\\", "\\midrule"]
    for p in pontos:
        cells = [p['rotulo_tex'], _pct(p['recall_severa']), str(p['severas_nao_sinalizadas']),
                 _pct(p['fa_severa']), _pct(p['fa_toxico'])]
        if p.get('destaque'):
            cells = [f"\\textbf{{{c}}}" for c in cells]
        out.append(" & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(out)


def pontos_operacao(y, P, tau_a, tau_b, llm):
    pontos = [dict(com_ic(taxas_argmax(y, P)), modelo='BERTimbau (argmax)',
                   rotulo_tex='BERTimbau (\\emph{argmax})', escolhido_no='—')]
    if tau_a == tau_b:
        regras = [('a+b', tau_a, f'BERTimbau ($\\tau = {tau_a:.2f}$, chosen on OOF, both rules)')]
    else:
        regras = [('a', tau_a, f'BERTimbau ($\\tau = {tau_a:.2f}$, chosen on OOF, FA budget)'),
                  ('b', tau_b, f'BERTimbau ($\\tau = {tau_b:.2f}$, chosen on OOF, max.\\ $F_2$)')]
    for regra, t, rot in regras:
        pontos.append(dict(com_ic(taxas_limiar(y, P, t)),
                           modelo=f'BERTimbau (tau = {t:.2f}, escolhido nas OOF, regra {regra})',
                           rotulo_tex=rot, escolhido_no='OOF', regra=regra,
                           destaque=regra in ('a', 'a+b')))
    if llm:
        m = np.asarray(llm['confusion_matrix'])
        n_sev, n_nen = int(m[2].sum()), int(m[0].sum())
        p = {
            'modelo': llm['nome'], 'rotulo_tex': _tex_escape(llm['nome']), 'escolhido_no': '—',
            'recall_severa': _div(m[2, 2], n_sev),
            'severas_nao_sinalizadas': int(n_sev - m[2, 2]),
            'severas_como_nenhuma': int(m[2, 0]),
            'fa_severa': _div(m[0, 2], n_nen),
            'fa_toxico': _div(n_nen - m[0, 0], n_nen),
            'n_severa': n_sev, 'n_nenhuma': n_nen,
            'n_severas_sinalizadas': int(m[2, 2]),
            'n_nenhuma_sinalizadas_severa': int(m[0, 2]),
            'n_nenhuma_sinalizadas_toxico': int(n_nen - m[0, 0]),
            'segundos_por_frase': llm.get('segundos_por_frase'),
        }
        pontos.append(com_ic(p))
    return pontos


# =============================================================================
# Figuras
# =============================================================================

def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'font.size': 9.5, 'axes.titlesize': 10, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8.5,
        'axes.edgecolor': EIXO, 'axes.linewidth': 0.8, 'axes.labelcolor': INK_SUAVE,
        'axes.grid': True, 'grid.color': GRADE, 'grid.linewidth': 0.6, 'grid.linestyle': '-',
        'text.color': INK, 'xtick.color': INK_SUAVE, 'ytick.color': INK_SUAVE,
        'xtick.major.size': 0, 'ytick.major.size': 0, 'legend.frameon': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })
    return plt


def _finalizar(plt, fig, axes, caminho):
    for ax in np.atleast_1d(axes).ravel():
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color(EIXO)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close(fig)


def fig_pr(caminho, pr_te, pr_oof, marcos):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.plot(pr_oof['recall'], pr_oof['precisao'], color=COR_OOF, lw=1.5,
            drawstyle='steps-post',
            label=_t('Out-of-fold, training split', 'Fora do fold, partição de treino')
            + f" (AP = {_n(pr_oof['average_precision'], 3)})")
    ax.plot(pr_te['recall'], pr_te['precisao'], color=COR_TESTE, lw=1.7,
            drawstyle='steps-post',
            label=_t('Held-out test', 'Teste retido')
            + f" (AP = {_n(pr_te['average_precision'], 3)})")
    ax.axhline(pr_te['prevalencia_severa'], color=INK_SUAVE, lw=0.8, ls=':',
               label=_t('No skill (test prevalence', 'Sem habilidade (prev. no teste')
               + f" {_n(pr_te['prevalencia_severa'], 2)})")
    for m in marcos:
        ax.plot(m['recall'], m['precisao'], m['marker'], ms=7, color=INK,
                markerfacecolor=m.get('face', INK), markeredgewidth=1.1, linestyle='none',
                label=m['rotulo'])
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    _marcas_unitarias(ax)
    ax.set_xlabel(_t('Recall (Severe)', 'Revocação (Severa)'))
    ax.set_ylabel(_t('Precision (Severe)', 'Precisão (Severa)'))
    # As curvas ocupam o alto do gráfico e a linha sem habilidade fica embaixo: a
    # legenda vai no meio, à esquerda, longe das duas.
    ax.legend(loc='center left', bbox_to_anchor=(0.0, 0.52))
    _finalizar(plt, fig, ax, caminho)


def fig_limiar(caminho, curva_te, curva_oof, tau_a, tau_b, teto):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), sharex=True)
    tau = [l['tau'] for l in curva_te]
    series = (('recall_severa', axes[0], _t('Severe recall', 'Revocação da Severa')),
              ('fa_severa', axes[1], _t('False alarm on None (flagged as Severe)',
                                        'Falso alarme em Nenhuma\n(sinalizada como Severa)')))
    for chave, ax, rot in series:
        ax.plot(tau, [100 * l[chave] for l in curva_oof], color=COR_OOF, lw=1.5,
                label=_t('Out-of-fold, training split', 'Fora do fold, partição de treino'))
        ax.plot(tau, [100 * l[chave] for l in curva_te], color=COR_TESTE, lw=1.7,
                label=_t('Held-out test', 'Teste retido'))
        ax.set_ylabel(f'{rot} (%)')
        ax.set_xlabel(_t(r'Threshold $\tau$ on $P(\mathrm{Severe})$ (log scale)',
                         r'Limiar $\tau$ sobre $P(\mathrm{Severa})$ (escala log)'))
        for t, ls in ((tau_a, '--'), (tau_b, ':')):
            ax.axvline(t, color=INK_SUAVE, lw=0.9, ls=ls)
    axes[1].axhline(100 * teto, color=COR_OOF, lw=0.8, ls='-.',
                    label=_t(f'OOF budget ceiling (argmax + {ORCAMENTO_FA_PP:g} pp)',
                             f'Teto do orçamento OOF (argmax + {_g(ORCAMENTO_FA_PP)} p.p.)'))
    # Escala log em tau: o compromisso acontece todo abaixo de ~0,3, que numa escala
    # linear fica espremido no canto esquerdo.
    marcas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.99]
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlim(GRADE_TAU[0], GRADE_TAU[-1])
        ax.set_xticks(marcas)
        ax.set_xticklabels([_g(m) for m in marcas])
        ax.minorticks_off()
    # Rótulos das linhas verticais no painel da esquerda.
    y0, y1 = axes[0].get_ylim()
    axes[0].text(tau_a, y0 + 0.04 * (y1 - y0),
                 _t(' OOF budget', ' orçamento OOF') + f'\n τ={_n(tau_a, 2)}', fontsize=7.5,
                 color=INK_SUAVE, va='bottom', ha='left')
    if tau_b != tau_a:
        axes[0].text(tau_b, y0 + 0.30 * (y1 - y0),
                     _t(' OOF max F2', ' F2 máx. OOF') + f'\n τ={_n(tau_b, 2)}', fontsize=7.5,
                     color=INK_SUAVE, va='bottom', ha='left')
    axes[0].set_title(_t('(a) Severe recall', '(a) Revocação da Severa'), loc='left')
    axes[1].set_title(_t('(b) False alarm', '(b) Falso alarme'), loc='left')
    # Legenda fora dos painéis: o formato das curvas muda a cada rodada, e dentro do
    # eixo ela acabava cobrindo a queda do falso alarme.
    alcas, rotulos = axes[1].get_legend_handles_labels()
    fig.legend(alcas, rotulos, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.07))
    _finalizar(plt, fig, axes, caminho)


def fig_calibracao(caminho, cal_oof, cal_te, cal_te_ts, T):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4))
    paineis = (('bins_top_label', 'ece_top_label',
                _t('(a) Top label (predicted class)', '(a) Rótulo do topo (classe predita)'),
                _t('Mean confidence of the predicted class', 'Confiança média da classe predita')),
               ('bins_severa', 'ece_severa',
                _t('(b) Severe class (one-vs-rest)', '(b) Classe Severa (um contra todos)'),
                _t(r'Mean predicted $P(\mathrm{Severe})$', r'$P(\mathrm{Severa})$ predita média')))
    series = ((cal_oof, COR_OOF, 's',
               _t('Out-of-fold, uncalibrated', 'Fora do fold (OOF), sem calibração'), 'OOF'),
              (cal_te, COR_TESTE, 'o', _t('Test, uncalibrated', 'Teste, sem calibração'),
               _t('test', 'teste')),
              (cal_te_ts, COR_CALIBRADO, '^',
               _t('Test, temperature-scaled', 'Teste, com escala de temperatura')
               + f' (T = {_n(T, 2)})',
               _t('test, scaled', 'teste, com escala')))
    from matplotlib.lines import Line2D
    for (chave_bins, chave_ece, titulo, rot_x), ax in zip(paineis, axes):
        ax.plot([0, 1], [0, 1], color=INK_SUAVE, lw=0.8, ls='--')
        for cal, cor, mk, rot, _ in series:
            n_total = sum(x['n'] for x in cal[chave_bins])
            b = [x for x in cal[chave_bins] if x['n'] >= MIN_N_BIN_FIGURA]
            xs = [x['confianca_media'] for x in b]
            ys = [x['frequencia_observada'] for x in b]
            ax.plot(xs, ys, color=cor, lw=1.1, alpha=0.8)
            # Área do marcador proporcional à fração das frases que cai no bin.
            ax.scatter(xs, ys, s=[8 + 100 * x['n'] / n_total for x in b], color=cor,
                       marker=mk, zorder=3, edgecolor='white', linewidth=0.5)
        ax.text(0.03, 0.97, "ECE\n" + "\n".join(f"{curto}: {_n(cal[chave_ece], 3)}"
                                                for cal, _, _, _, curto in series),
                transform=ax.transAxes, va='top', ha='left', fontsize=7.5, color=INK_SUAVE)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        _marcas_unitarias(ax)
        ax.set_aspect('equal')
        ax.set_title(titulo, loc='left')
        ax.set_xlabel(rot_x)
        ax.set_ylabel(_t('Observed frequency', 'Frequência observada'))
    alcas = [Line2D([], [], color=INK_SUAVE, lw=0.8, ls='--',
                    label=_t('Perfect calibration', 'Calibração perfeita'))]
    alcas += [Line2D([], [], color=cor, marker=mk, ms=5, lw=1.1, label=rot)
              for _, cor, mk, rot, _ in series]
    fig.legend(handles=alcas, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.06),
               fontsize=8)
    fig.text(0.5, -0.11,
             _t(f"{N_BINS_ECE} equal-width bins; marker area proportional to the "
                f"share of sentences in the bin; bins with fewer than {MIN_N_BIN_FIGURA} "
                "sentences are not drawn but enter the ECE.",
                f"{N_BINS_ECE} bins de mesma largura; área do marcador proporcional à fração "
                f"das frases no bin; bins com menos de {MIN_N_BIN_FIGURA} frases não são "
                "desenhados, mas entram no ECE."),
             ha='center', fontsize=7.5, color=INK_SUAVE)
    _finalizar(plt, fig, axes, caminho)


# =============================================================================
# JSON
# =============================================================================

def _limpar(o):
    """Converte tipos numpy e troca NaN/inf por None (JSON estrito)."""
    if isinstance(o, dict):
        return {str(k): _limpar(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_limpar(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_limpar(v) for v in o.tolist()]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, Path):
        return str(o)
    return o


def _dentro(filho: Path, pai: Path) -> bool:
    try:
        filho.resolve().relative_to(pai.resolve())
        return True
    except ValueError:
        return False


# =============================================================================
# Principal
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description='Limiar OOF, curva PR, calibração e pesos do score')
    ap.add_argument('--resultados', type=Path, default=None,
                    help='pasta de resultados de onde ler as entradas (padrão: a da rodada '
                         'atual, experimento.dir_resultados()); aceita a pasta da rodada ou '
                         'a subpasta modelo_final/')
    ap.add_argument('--saida', type=Path, default=None,
                    help='pasta para o JSON; figuras e tabelas vão para <saida>/latex/. '
                         'Padrão: dir_resultados() e dir_latex() da rodada atual')
    ap.add_argument('--ts', default=None,
                    help='timestamp do treino final a usar (padrão: o mais recente)')
    ap.add_argument('--sem-textos', action='store_true',
                    help='não busca o texto das frases no corpus (item m6)')
    ap.add_argument('--idioma', choices=['en', 'pt'], default='en',
                    help='idioma dos rótulos das figuras (pt: sufixo -pt nos arquivos; '
                         'exige --saida); os números do JSON e as tabelas não mudam')
    args = ap.parse_args()
    if args.idioma != 'en' and not args.saida:
        sys.exit("--idioma pt exige --saida: sem ela o script gravaria um JSON novo e "
                 "reescreveria as tabelas na pasta da rodada.")
    global IDIOMA
    IDIOMA = args.idioma

    tag_atual = experimento.tag()
    pasta_res = (args.resultados.resolve() if args.resultados
                 else experimento.dir_resultados())
    if pasta_res.name == 'modelo_final':
        pasta_res = pasta_res.parent
    ts, arq_oof, arq_teste, arq_json = localizar_entradas(pasta_res, args.ts)
    meta = {}
    if arq_json:
        with open(arq_json, encoding='utf-8') as f:
            meta = json.load(f)
    tag_entradas = (meta.get('experimento') or {}).get('tag')
    mesma_rodada = (tag_entradas == tag_atual) if tag_entradas else (args.resultados is None)

    # --- destinos (nunca o latex/ do artigo; validação nunca na pasta de entrada)
    if args.saida:
        saida = args.saida.resolve()
        if _dentro(saida, experimento.ARTIGO_ROOT / 'latex'):
            sys.exit("--saida não pode ficar dentro de latex/: as tabelas entram no "
                     "manuscrito por cópia deliberada.")
        if not mesma_rodada and _dentro(saida, pasta_res):
            sys.exit("--saida não pode ficar dentro da pasta das entradas de outra rodada.")
        dir_json, dir_fig, dir_tab = saida, saida / 'latex' / 'figuras', saida / 'latex' / 'tabelas'
        for d in (dir_json, dir_fig, dir_tab):
            d.mkdir(parents=True, exist_ok=True)
    else:
        if not mesma_rodada:
            sys.exit(f"As entradas são da rodada {tag_entradas!r}, não da atual ({tag_atual!r}): "
                     "use --saida para não misturar resultados.")
        dir_json = experimento.dir_resultados()
        dir_fig = experimento.dir_latex('figuras')
        dir_tab = experimento.dir_latex('tabelas')

    d = carregar_probabilidades(arq_oof, arq_teste, meta.get('semente_referencia'))
    y_oof, P_oof = d['y_oof'], d['P_oof']
    y_te, ref = d['y_teste'], d['semente_ref']
    P_te = d['P_teste'][ref]
    print(f"Entradas: {arq_oof.parent} (ts {ts}; rodada das entradas: {tag_entradas}; "
          f"atual: {tag_atual})")
    print(f"OOF: {len(y_oof)} frases ({d['oof_descartadas_nan']} sem probabilidade descartadas) | "
          f"teste: {len(y_te)} frases | semente de referência: {ref} | sementes: {d['sementes']}")

    # ------------------------------------------------------------------ 1. M6e
    curva_oof = curva_limiar(y_oof, P_oof)
    curva_te = curva_limiar(y_te, P_te)
    arg_oof, arg_te = taxas_argmax(y_oof, P_oof), taxas_argmax(y_te, P_te)
    tau_a, teto = escolher_tau_orcamento(curva_oof, arg_oof['fa_severa'])
    tau_b = escolher_tau_fbeta(curva_oof)

    def bloco_regra(tau, descricao, extra=None):
        por_semente = [dict(taxas_limiar(y_te, d['P_teste'][s], tau), semente=s)
                       for s in d['sementes']]
        b = {'descricao': descricao, 'tau': tau, 'escolhido_em': 'OOF',
             'oof': taxas_limiar(y_oof, P_oof, tau),
             'teste': com_ic(taxas_limiar(y_te, P_te, tau)),
             'teste_por_semente': por_semente,
             'teste_resumo_sementes': resumo_sementes(por_semente)}
        b.update(extra or {})
        return b

    arg_por_semente = [dict(taxas_argmax(y_te, d['P_teste'][s]), semente=s) for s in d['sementes']]
    limiar_oof = {
        'escolhido_em': ('predições out-of-fold da partição de treino (folds agrupados por '
                         'vídeo, semente de referência); o teste retido só é usado para '
                         'reportar'),
        'regras_declaradas_a_priori': True,
        'regra_principal': REGRA_PRINCIPAL,
        'grade_tau': {'inicio': float(GRADE_TAU[0]), 'fim': float(GRADE_TAU[-1]), 'passo': 0.01},
        'argmax': {'oof': arg_oof, 'teste': com_ic(arg_te), 'teste_por_semente': arg_por_semente,
                   'teste_resumo_sementes': resumo_sementes(arg_por_semente)},
        'regras': {
            'a_orcamento_fa': bloco_regra(
                tau_a,
                ("menor tau da grade cujo falso alarme OOF (Nenhuma sinalizada como Severa, "
                 "P(Severa) > tau) fique <= falso alarme OOF do argmax + orçamento"),
                {'orcamento_pp': ORCAMENTO_FA_PP, 'fa_argmax_oof': arg_oof['fa_severa'],
                 'teto_fa_oof': teto}),
            'b_max_f2': bloco_regra(
                tau_b,
                "tau da grade que maximiza o F2 (beta = 2) da classe Severa nas OOF; empate -> menor tau",
                {'beta': BETA_F}),
        },
    }
    # Comparação exploratória: as mesmas regras aplicadas ao próprio teste.
    tau_a_te, teto_te = escolher_tau_orcamento(curva_te, arg_te['fa_severa'])
    tau_b_te = escolher_tau_fbeta(curva_te)
    exploratorio = {
        'aviso': ('EXPLORATÓRIO: limiares escolhidos no próprio teste, só para mostrar o '
                  'otimismo da escolha no teste; não usar como ponto de operação'),
        'a_orcamento_fa': {'tau': tau_a_te, 'teto_fa_teste': teto_te,
                           'teste': taxas_limiar(y_te, P_te, tau_a_te)},
        'b_max_f2': {'tau': tau_b_te, 'teste': taxas_limiar(y_te, P_te, tau_b_te)},
    }

    # ------------------------------------------------------------------- 2. m7
    pr_te, pr_oof = curva_pr(y_te, P_te), curva_pr(y_oof, P_oof)
    t50 = taxas_limiar(y_te, P_te, 0.50)
    argmax_coincide_050 = (t50['n_severas_sinalizadas'] == arg_te['n_severas_sinalizadas'] and
                           t50['n_nenhuma_sinalizadas_severa'] == arg_te['n_nenhuma_sinalizadas_severa'])
    linhas_tab = linhas_tabela_limiar(y_oof, P_oof, y_te, P_te, tau_a, tau_b)

    llm = carregar_llm(pasta_res, len(y_te))
    if llm and not llm['mesmo_n_do_teste']:
        print(f"AVISO: a matriz do LLM ({llm['fonte']}) soma {llm['n_amostras']} frases, "
              f"o teste tem {len(y_te)}.")
    pontos = pontos_operacao(y_te, P_te, tau_a, tau_b, llm)

    # ------------------------------------------------------------------- 3. m6
    textos, motivo_sem_textos = None, None
    if args.sem_textos:
        motivo_sem_textos = '--sem-textos'
    elif not mesma_rodada:
        motivo_sem_textos = (f"entradas da rodada {tag_entradas!r}: os id_sentenca não "
                             f"correspondem ao corpus atual ({tag_atual!r})")
    else:
        textos, motivo_sem_textos = carregar_textos(d['ids_teste'], y_te)
    if motivo_sem_textos:
        print(f"Textos das frases (m6) omitidos: {motivo_sem_textos}")
    perdidas = severas_preditas_nenhuma(d['ids_teste'], y_te, P_te, textos, tau_a)
    perdidas['textos_incluidos'] = textos is not None
    perdidas['motivo_textos_omitidos'] = motivo_sem_textos

    # ------------------------------------------------------------------ 4. M7a
    temp = ajustar_temperatura(y_oof, P_oof)
    T = temp['T']
    P_oof_ts, P_te_ts = aplicar_temperatura(P_oof, T), aplicar_temperatura(P_te, T)
    cal = {
        'n_bins': N_BINS_ECE,
        'definicoes': {
            'bins': 'largura igual em [0, 1]: [0, 1/15], (1/15, 2/15], ..., (14/15, 1]',
            'ece_top_label': 'sum_b (n_b/N) |acurácia_b - confiança média_b|, confiança = max_k p_k',
            'ece_por_classe': 'para cada classe k: sum_b (n_b/N) |freq(y=k)_b - média(p_k)_b|',
            'brier_multiclasse': 'média sobre frases de sum_k (p_k - 1[y=k])^2 (intervalo 0 a 2)',
            'nll': 'média de -log p_y',
            'figura': (f'fig-calibracao.png desenha só bins com >= {MIN_N_BIN_FIGURA} frases; '
                       'o ECE usa todos os bins'),
        },
        'antes': {'oof': metricas_calibracao(y_oof, P_oof),
                  'teste': metricas_calibracao(y_te, P_te)},
        'temperatura': dict(temp, ajustada_em='OOF',
                            premissa=('as OOF vêm de 5 modelos de fold (semente de referência) e '
                                      'o teste do modelo final treinado em todo o desenvolvimento; '
                                      'aplicar ao teste a T das OOF supõe calibração semelhante')),
        'depois': {'oof_dentro_da_amostra': metricas_calibracao(y_oof, P_oof_ts),
                   'teste': metricas_calibracao(y_te, P_te_ts)},
        'teste_por_semente': [
            {'semente': s,
             'antes': metricas_calibracao(y_te, d['P_teste'][s], com_bins=False),
             'depois': metricas_calibracao(y_te, aplicar_temperatura(d['P_teste'][s], T),
                                           com_bins=False)}
            for s in d['sementes']],
        'efeito_no_score': {
            'score': 'P = 1 - (0.5 * p_leve + 1.0 * p_severa)',
            'nota_auc': 'positivo = classe menos tóxica; score maior = mais seguro',
            'teste': efeito_temperatura_no_score(y_te, P_te, P_te_ts),
            'oof_dentro_da_amostra': efeito_temperatura_no_score(y_oof, P_oof, P_oof_ts),
        },
    }

    # ------------------------------------------------------------------ 5. M7b
    sens = {
        'nota': ('pesos ordinais e equidistantes (Nenhuma 0, Leve 0,5, Severa 1), fixados a '
                 'priori, antes de qualquer resultado; aqui só se varia o peso da Leve, com '
                 'Nenhuma 0 e Severa 1 fixos. A justificativa textual é do autor.'),
        'pesos_leve': list(PESOS_LEVE_SENSIBILIDADE),
        'peso_nenhuma': 0.0, 'peso_severa': 1.0,
        'score': 'P = 1 - (w_leve * p_leve + 1.0 * p_severa)',
        'nota_auc': 'positivo = classe menos tóxica; score maior = mais seguro',
        'teste': sensibilidade_pesos(y_te, P_te),
        'oof': sensibilidade_pesos(y_oof, P_oof),
    }

    # --------------------------------------------------------------- figuras
    t_a, t_b = taxas_limiar(y_te, P_te, tau_a), taxas_limiar(y_te, P_te, tau_b)
    marcos = [{'recall': arg_te['recall_severa'], 'precisao': arg_te['precisao_severa'],
               'marker': 'o', 'rotulo': _t('Test, argmax', 'Teste, argmax')},
              {'recall': t_a['recall_severa'], 'precisao': t_a['precisao_severa'],
               'marker': 'D', 'face': 'white',
               'rotulo': _t('Test', 'Teste') + f', τ = {_n(tau_a, 2)} '
                         + _t('(OOF, FA budget)', '(OOF, orçamento de FA)')}]
    if tau_b != tau_a:
        marcos.append({'recall': t_b['recall_severa'], 'precisao': t_b['precisao_severa'],
                       'marker': 's', 'face': 'white',
                       'rotulo': _t('Test', 'Teste') + f', τ = {_n(tau_b, 2)} '
                                 + _t('(OOF, max F2)', '(OOF, F2 máximo)')})
    figuras = {
        'fig-pr-severa.png': lambda c: fig_pr(c, pr_te, pr_oof, marcos),
        'fig-limiar-severa.png': lambda c: fig_limiar(c, curva_te, curva_oof, tau_a, tau_b, teto),
        'fig-calibracao.png': lambda c: fig_calibracao(c, cal['antes']['oof'], cal['antes']['teste'],
                                                       cal['depois']['teste'], T),
    }
    caminhos_fig = []
    for nome, fazer in figuras.items():
        c = dir_fig / (nome if IDIOMA == 'en' else nome.replace('.png', '-pt.png'))
        fazer(c)
        caminhos_fig.append(str(c))

    # --------------------------------------------------------------- tabelas
    caminhos_tab = []
    for nome, conteudo in (('tab-limiar.tex', tex_limiar(linhas_tab, argmax_coincide_050)),
                           ('tab-pontos-operacao.tex', tex_pontos(pontos))):
        c = dir_tab / nome
        with open(c, 'w', encoding='utf-8') as f:
            f.write(conteudo)
        caminhos_tab.append(str(c))

    # ------------------------------------------------------------------ JSON
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    resultado = {
        'script': 'app/nlp/evaluation/scripts/analisar_calibracao_limiar.py',
        'gerado_em': stamp,
        'experimento_atual': experimento.descricao(),
        'entradas': {
            'pasta': str(arq_oof.parent), 'timestamp_treino_final': ts,
            'arquivo_oof': arq_oof.name, 'arquivo_teste': arq_teste.name,
            'arquivo_modelo_final': arq_json.name if arq_json else None,
            'experimento_das_entradas': meta.get('experimento'),
            'mesma_rodada_que_a_atual': bool(mesma_rodada),
            'arquivo_llm': llm['fonte'] if llm else None,
        },
        'semente_referencia': ref, 'sementes_teste': d['sementes'],
        'n': {'oof': int(len(y_oof)), 'oof_descartadas_nan': d['oof_descartadas_nan'],
              'teste': int(len(y_te)),
              'oof_por_classe': {c: int((y_oof == k).sum()) for k, c in enumerate(CLASSES)},
              'teste_por_classe': {c: int((y_te == k).sum()) for k, c in enumerate(CLASSES)}},
        'convencoes': {
            'regra_sinalizacao_severa': 'P(Severa) > tau (desigualdade estrita)',
            'fa_severa': 'fração das frases Nenhuma sinalizadas como Severa',
            'fa_toxico': ('fração das frases Nenhuma sinalizadas como tóxicas em qualquer grau; '
                          'no limiar: argmax != Nenhuma OU P(Severa) > tau'),
            'severas_nao_sinalizadas': 'Severas com P(Severa) <= tau (no argmax: argmax != Severa)',
            'severas_como_nenhuma': 'Severas que não são sinalizadas em grau nenhum',
            'pesos_score': list(PESOS),
            'probabilidades_teste': f'semente de referência ({ref}), salvo onde indicado',
        },
        'limiar_oof': limiar_oof,
        'exploratorio_escolha_no_teste': exploratorio,
        'curva_limiar': {'teste': curva_te, 'oof': curva_oof},
        'curva_pr_severa': {'teste': pr_te, 'oof': pr_oof},
        'argmax_coincide_tau_050_no_teste': bool(argmax_coincide_050),
        'tabela_limiar_linhas': linhas_tab,
        'pontos_operacao': pontos,
        'llm': llm,
        'severas_preditas_nenhuma': perdidas,
        'calibracao': cal,
        'sensibilidade_pesos': sens,
        'figuras': caminhos_fig,
        'tabelas': caminhos_tab,
    }
    destino = dir_json / f'calibracao_limiar_{stamp}.json'
    with open(destino, 'w', encoding='utf-8') as f:
        json.dump(_limpar(resultado), f, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------- resumo
    def pr(x):
        return f"{100 * x:.1f}%" if x is not None else "-"

    print("\n--- limiar escolhido nas OOF (M6e) ---")
    print(f"  argmax OOF: FA(Severa)={pr(arg_oof['fa_severa'])} -> teto da regra (a) = {pr(teto)}")
    for nome, b in limiar_oof['regras'].items():
        t = b['teste']
        print(f"  {nome}: tau={b['tau']:.2f} | OOF recall={pr(b['oof']['recall_severa'])} "
              f"FA={pr(b['oof']['fa_severa'])} | TESTE recall={pr(t['recall_severa'])} "
              f"não sinalizadas={t['severas_nao_sinalizadas']} FA(Severa)={pr(t['fa_severa'])} "
              f"FA(qualquer)={pr(t['fa_toxico'])}")
    print(f"  argmax TESTE: recall={pr(arg_te['recall_severa'])} perdidas="
          f"{arg_te['severas_nao_sinalizadas']} FA(Severa)={pr(arg_te['fa_severa'])} "
          f"FA(qualquer)={pr(arg_te['fa_toxico'])}")
    print(f"  [exploratório, escolhido no teste] (a) tau={tau_a_te:.2f}  (b) tau={tau_b_te:.2f}")
    print(f"\n--- curva PR da Severa (m7) --- AP teste={pr_te['average_precision']:.4f} "
          f"AP OOF={pr_oof['average_precision']:.4f}")
    print(f"  linhas da tab-limiar: "
          f"{['argmax' if l['tau'] is None else l['tau'] for l in linhas_tab]}")
    print(f"\n--- Severas preditas como Nenhuma (m6): {perdidas['n']} de {perdidas['n_severa_teste']} ---")
    for f_ in perdidas['frases']:
        txt = (f_['texto'] or '')[:70]
        print(f"  id={f_['id_sentenca']} score={f_['score']:.3f} p(Severa)={f_['p_severa']:.3f} {txt}")
    a, dp = cal['antes'], cal['depois']
    print(f"\n--- calibração (M7a) --- T={T:.3f} (ajustada nas OOF; no limite: {temp['no_limite']})")
    for nome, ca, cd in (('OOF', a['oof'], dp['oof_dentro_da_amostra']), ('teste', a['teste'], dp['teste'])):
        print(f"  {nome}: ECE top {ca['ece_top_label']:.4f}->{cd['ece_top_label']:.4f} | "
              f"ECE Severa {ca['ece_severa']:.4f}->{cd['ece_severa']:.4f} | "
              f"Brier {ca['brier_multiclasse']:.4f}->{cd['brier_multiclasse']:.4f} | "
              f"NLL {ca['nll']:.4f}->{cd['nll']:.4f}")
    e = cal['efeito_no_score']['teste']
    print(f"  score no teste: AUC(Nenhuma x tóxica) {e['antes']['auc_nenhuma_vs_toxica']:.4f}->"
          f"{e['depois']['auc_nenhuma_vs_toxica']:.4f}; medianas antes "
          f"{[round(e['antes']['por_classe'][c]['mediana'], 3) for c in CLASSES]} depois "
          f"{[round(e['depois']['por_classe'][c]['mediana'], 3) for c in CLASSES]}; "
          f"Spearman {e['spearman_antes_depois']:.4f}")
    print("\n--- sensibilidade ao peso da Leve (M7b) ---")
    for conj in ('teste', 'oof'):
        for w, r in sens[conj]['por_peso_leve'].items():
            print(f"  {conj} w_leve={w}: AUC(N x tox)={r['auc_nenhuma_vs_toxica']:.4f} "
                  f"AUC(N x Sev)={r['auc_nenhuma_vs_severa']:.4f} medianas="
                  f"{[round(r['por_classe'][c]['mediana'], 3) for c in CLASSES]}")
        print(f"  {conj} Spearman: { {k: round(v, 4) for k, v in sens[conj]['spearman_entre_pesos'].items()} }")
    print(f"\nJSON: {destino}")
    for c in caminhos_fig + caminhos_tab:
        print(f"  {c}")


if __name__ == '__main__':
    main()
