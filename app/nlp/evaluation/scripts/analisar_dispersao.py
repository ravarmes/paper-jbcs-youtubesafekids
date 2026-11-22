"""
Dispersão comparável entre o BERTimbau e as baselines, e a Tabela 3 do artigo.

Responde aos itens M3a e M3c da revisão do JBCS:

M3a -- a Tabela 3 da versão submetida misturava dois tipos de dispersão: o
    BERTimbau levava o desvio ENTRE SEMENTES (estágio 3 da busca) e as baselines
    o desvio ENTRE FOLDS. Aqui todos os modelos são reportados com o mesmo tipo
    -- desvio-padrão amostral (ddof=1) entre os mesmos 5 folds agrupados por
    vídeo -- e o desvio entre sementes do BERTimbau vai à parte. A margem sobre a
    baseline mais forte é expressa em unidades desse desvio entre folds.

M3c -- o resultado no teste retido do BERTimbau passa a ser a média ± desvio
    entre as 5 sementes do treino final (modelo_final_*.json), com a semente de
    referência (usada nas análises a jusante) indicada à parte, junto dos dados,
    da semente e da época restaurada de cada modelo final.

Entradas (as mais recentes de cada padrão, na pasta de resultados da rodada ou
em --resultados):
    toxicity_gridsearch/gridsearch_optimized_search_*.json   (obrigatório)
    modelo_final/modelo_final_*.json                          (obrigatório)
    baselines_toxicity_*.json                                 (baselines fixas)
    baselines_ajustados_*.json                                (baselines ajustadas, M3b)
    logs/final.log                                            (época restaurada; opcional)

Saídas:
    dispersao_<timestamp>.json  em experimento.dir_resultados() (ou --saida)
    tab-baselines.tex           em experimento.dir_latex('tabelas') (ou --saida)

Uso (de dentro de src/). Rode pelo CAMINHO do arquivo: `python -m` importaria o
pacote app.nlp.evaluation, cujo __init__ carrega o torch (~2 GB de memória
comprometida) sem necessidade -- este script só lê JSON.
    python app/nlp/evaluation/scripts/analisar_dispersao.py
    python app/nlp/evaluation/scripts/analisar_dispersao.py \\
        --resultados app/nlp/evaluation/results/video-original --saida <pasta>
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp import experimento  # noqa: E402

CLASSES = ['Nenhuma', 'Leve', 'Severa']
NOME_BERT = 'BERTimbau (fine-tuned)'


# ----------------------------------------------------------------------------
# Leitura
# ----------------------------------------------------------------------------

def _mais_recente(padrao):
    arquivos = glob.glob(padrao)
    return max(arquivos, key=os.path.getmtime) if arquivos else None


def _carregar(pasta, padrao, obrigatorio=True):
    caminho = _mais_recente(os.path.join(pasta, padrao))
    if caminho is None:
        if obrigatorio:
            raise FileNotFoundError(f"Nenhum arquivo {padrao} em {pasta}")
        return None, None
    with open(caminho, encoding='utf-8') as f:
        return json.load(f), os.path.basename(caminho)


def _dp(valores, ddof=1):
    v = np.asarray(valores, dtype=float)
    return float(v.std(ddof=ddof)) if len(v) > ddof else None


# ----------------------------------------------------------------------------
# BERTimbau
# ----------------------------------------------------------------------------

def dispersao_bertimbau(grid):
    """
    Desvio entre folds e desvio entre sementes da configuração vencedora.

    A busca em 3 estágios guarda, para a vencedora:
      - estágio 2 (semente 42): o F1 de cada um dos 5 folds (fold_metrics);
      - estágio 3 (3 sementes): por semente, a média e o desvio entre folds
        (pandas .std(), ddof=1) -- os F1 por fold do estágio 3 não são gravados.
    O F1 de CV reportado (best_config.mean_f1) é a média, entre sementes, das
    médias por fold do estágio 3. O desvio entre folds comparável ao das
    baselines é, então, a média entre sementes do desvio entre folds de cada
    semente; o desvio entre sementes (best_config.std_f1, np.std com ddof=0) é
    outra grandeza e vai em campo próprio.
    """
    melhor = grid.get('best_config')
    if not melhor:
        raise ValueError("gridsearch sem best_config")
    config = melhor['config']

    estagio2 = None
    for r in (grid.get('stage2') or {}).get('results', []):
        if r.get('config') == config:
            validos = [m for m in r.get('fold_metrics', []) if 'error' not in m]
            validos.sort(key=lambda m: m.get('fold', 0))
            f1s = [float(m['eval_f1']) for m in validos]
            estagio2 = {
                'semente': 42,
                'media': float(np.mean(f1s)) if f1s else r.get('mean_f1'),
                'dp_entre_folds': _dp(f1s),
                'dp_entre_folds_gravado': r.get('std_f1'),
                'por_fold': f1s,
                'folds': [m.get('fold') for m in validos],
            }
            break

    por_semente = melhor.get('por_semente') or []
    cv = {'media': float(melhor['mean_f1'])}
    if por_semente:
        medias = [float(s['mean_f1']) for s in por_semente]
        dps = [float(s['std_f1']) for s in por_semente if s.get('std_f1') is not None]
        cv.update({
            'media_tipo': 'média entre sementes das médias por fold (estágio 3)',
            'n_sementes': len(por_semente),
            'sementes': [s['seed'] for s in por_semente],
            'media_por_semente': medias,
            'dp_entre_folds_por_semente': dps,
            'dp_entre_folds': float(np.mean(dps)) if dps else None,
            'dp_entre_folds_tipo': 'média entre sementes do DP amostral (ddof=1) entre os 5 folds',
            'dp_entre_sementes': _dp(medias, ddof=1),
            'dp_entre_sementes_ddof0': _dp(medias, ddof=0),
            'dp_entre_sementes_gravado': melhor.get('std_f1'),
        })
    else:
        # Busca em 2 estágios (modo 'optimized'): uma semente só, e o std_f1
        # gravado já é o desvio entre folds.
        cv.update({
            'media_tipo': 'média entre folds (uma semente)',
            'n_sementes': 1,
            'dp_entre_folds': (estagio2 or {}).get('dp_entre_folds') or melhor.get('std_f1'),
            'dp_entre_folds_tipo': 'DP amostral (ddof=1) entre os 5 folds',
            'dp_entre_sementes': None,
        })

    n_stage1 = len((grid.get('stage1') or {}).get('results', []))
    return {
        'config': config,
        'cv': cv,
        'estagio2_semente42': estagio2,
        'n_configuracoes_busca': grid.get('n_amostras', n_stage1),
        'metodo_busca': grid.get('method'),
    }


def _epocas_do_log(caminho_log, n_sementes):
    """
    Época restaurada e tamanho do treino/dev interno de cada modelo final.

    O run_final treina primeiro as sementes, na ordem de `sementes`, e depois os
    5 folds out-of-fold; o log registra, para cada treino, a configuração (com a
    semente), o tamanho do treino e do dev interno, e a época cujos pesos foram
    restaurados. Só os n_sementes primeiros treinos são os modelos finais.
    """
    if not caminho_log or not os.path.exists(caminho_log):
        return None
    with open(caminho_log, encoding='utf-8', errors='replace') as f:
        texto = f.read()
    blocos = re.split(r"Sobrescrevendo configuração padrão com:", texto)[1:]
    saida = []
    for bloco in blocos[:n_sementes]:
        semente = re.search(r"'seed':\s*(\d+)", bloco)
        epoca = re.search(r"Pesos restaurados da época (\d+) \(eval_f1=([0-9.]+)\)", bloco)
        saida.append({
            'semente': int(semente.group(1)) if semente else None,
            'epoca_restaurada': int(epoca.group(1)) if epoca else None,
            'f1_dev_interno': float(epoca.group(2)) if epoca else None,
        })
    tamanhos = re.findall(r"Treino final: (\d+) amostras \| dev interno: (\d+)", texto)
    for item, (n_tr, n_dev) in zip(saida, tamanhos):
        item['n_treino'] = int(n_tr)
        item['n_dev_interno'] = int(n_dev)
    return saida


def teste_bertimbau(final, caminho_log=None):
    """Teste retido: média ± DP entre sementes, e a semente de referência à parte."""
    por_semente = final.get('teste_por_semente') or []
    if not por_semente:
        raise ValueError("modelo_final sem teste_por_semente")
    f1s = [float(m['f1_macro']) for m in por_semente]
    accs = [float(m['accuracy']) for m in por_semente]
    por_classe = np.array([[float(m['por_classe'][c]['f1-score']) for c in CLASSES]
                           for m in por_semente])
    ref = final.get('semente_referencia', por_semente[0]['seed'])
    m_ref = next((m for m in por_semente if m['seed'] == ref), por_semente[0])
    return {
        'n_sementes': len(f1s),
        'sementes': [m['seed'] for m in por_semente],
        'f1_macro_media': float(np.mean(f1s)),
        'f1_macro_dp_entre_sementes': _dp(f1s),
        'f1_macro_min': float(np.min(f1s)),
        'f1_macro_max': float(np.max(f1s)),
        'f1_macro_por_semente': dict(zip([str(m['seed']) for m in por_semente], f1s)),
        'accuracy_media': float(np.mean(accs)),
        'accuracy_dp_entre_sementes': _dp(accs),
        'f1_por_classe_media': dict(zip(CLASSES, por_classe.mean(axis=0).tolist())),
        'f1_por_classe_dp_entre_sementes': dict(zip(
            CLASSES, por_classe.std(axis=0, ddof=1).tolist() if len(f1s) > 1 else [None] * 3)),
        'semente_referencia': ref,
        'modelo_referencia': m_ref.get('modelo'),
        'f1_macro_semente_referencia': float(m_ref['f1_macro']),
        'accuracy_semente_referencia': float(m_ref['accuracy']),
        'dados_treino_final': {
            'descricao': ('conjunto de desenvolvimento inteiro (todas as frases fora do teste), '
                          'com um dev interno agrupado por vídeo separado para o early stopping '
                          '(escolha da época); mesma configuração e mesmos dados para todas as '
                          'sementes, que só mudam a inicialização e a ordem dos lotes'),
            'n_dev': final.get('n_dev'),
            'n_teste': final.get('n_teste'),
            'epocas_maximas': (final.get('config') or {}).get('epochs'),
            'por_semente_log': _epocas_do_log(caminho_log, len(por_semente)),
        },
        'oof_f1_macro': final.get('oof_f1_macro'),
    }


# ----------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------

def baselines_fixas(base):
    """Baselines de train_toxicity_baselines.py, com DP amostral entre folds."""
    k = int((base.get('methodology') or {}).get('cv_folds', 5))
    saida = {}
    for nome, cv in base['cross_validation_results'].items():
        if cv.get('f1_por_fold'):
            dp = _dp(cv['f1_por_fold'])
            convertido = False
        else:
            # JSONs anteriores só têm np.std (ddof=0); a conversão para ddof=1
            # é exata: s1 = s0 * sqrt(k / (k - 1)).
            dp = float(cv['f1_std']) * np.sqrt(k / (k - 1))
            convertido = True
        teste = base['test_results'][nome]
        saida[nome] = {
            'tipo': 'fixa',
            'selecao': 'fixed',
            'cv': {'media': float(cv['f1_mean']), 'dp_entre_folds': dp,
                   'por_fold': cv.get('f1_por_fold'),
                   'dp_convertido_de_ddof0': convertido},
            'teste_f1_macro': float(teste['f1']),
            'teste_accuracy': float(teste['accuracy']),
            'teste_f1_por_classe': teste.get('f1_per_class'),
        }
    return saida


def baselines_ajustadas(ajust):
    """Baselines de train_toxicity_baselines_ajustados.py (CV aninhada)."""
    saida = {}
    for chave, r in ajust['resultados'].items():
        an = r['cv_aninhada']['f1_macro']
        pl = r['cv_plana']['melhor_f1_macro']
        saida[r['nome']] = {
            'tipo': 'ajustada',
            'chave': chave,
            'selecao': f"nested CV, {r['n_configuracoes']} configs",
            'n_configuracoes': r['n_configuracoes'],
            'cv': {'media': float(an['media']), 'dp_entre_folds': float(an['dp_entre_folds']),
                   'por_fold': an['por_fold'], 'dp_convertido_de_ddof0': False},
            'cv_plana_melhor': {'media': float(pl['media']),
                                'dp_entre_folds': float(pl['dp_entre_folds']),
                                'por_fold': pl['por_fold'],
                                'config': r['cv_plana']['melhor_config']},
            'otimismo_selecao': float(pl['media'] - an['media']),
            'configs_por_fold_externo': [f['config_escolhida'] for f in r['cv_aninhada']['por_fold']],
            'teste_f1_macro': float(r['teste']['f1_macro']),
            'teste_accuracy': float(r['teste']['accuracy']),
            'teste_f1_por_classe': r['teste'].get('f1_per_class'),
            'config_final': r['teste'].get('config'),
        }
    return saida


# ----------------------------------------------------------------------------
# Margens
# ----------------------------------------------------------------------------

def _margem(bert_cv, bert_por_fold, base, campo_cv='cv'):
    cv_b = base[campo_cv]
    margem = bert_cv['media'] - cv_b['media']
    s_bert, s_base = bert_cv.get('dp_entre_folds'), cv_b.get('dp_entre_folds')
    out = {
        'cv_bertimbau': bert_cv['media'],
        'cv_baseline': cv_b['media'],
        'margem': float(margem),
        'dp_folds_bertimbau': s_bert,
        'dp_folds_baseline': s_base,
        'margem_em_dp_folds_bertimbau': float(margem / s_bert) if s_bert else None,
        'margem_em_dp_folds_baseline': float(margem / s_base) if s_base else None,
        'margem_em_dp_folds_combinado': (float(margem / np.sqrt((s_bert ** 2 + s_base ** 2) / 2))
                                         if s_bert and s_base else None),
    }
    # Diferença fold a fold: os dois modelos foram avaliados nos MESMOS folds,
    # então a variação de dificuldade entre folds se cancela. Usa os F1 por fold
    # do estágio 2 (semente 42), os únicos gravados por fold para o BERTimbau.
    if bert_por_fold and cv_b.get('por_fold') and len(bert_por_fold) == len(cv_b['por_fold']):
        d = np.asarray(bert_por_fold) - np.asarray(cv_b['por_fold'])
        out['pareado_por_fold'] = {
            'fonte_bertimbau': 'estágio 2 da busca (semente 42), F1 por fold',
            'diferencas': d.tolist(),
            'media': float(d.mean()),
            'dp': _dp(d),
            'media_em_dp': float(d.mean() / _dp(d)) if _dp(d) else None,
            'folds_bertimbau_melhor': int((d > 0).sum()),
            'n_folds': int(len(d)),
        }
    return out


def calcular_margens(bert, teste_bert, baselines):
    bert_cv = bert['cv']
    bert_por_fold = (bert.get('estagio2_semente42') or {}).get('por_fold')
    por_baseline = {}
    for nome, b in baselines.items():
        m = {'cv': _margem(bert_cv, bert_por_fold, b)}
        if 'cv_plana_melhor' in b:
            # Mesma regra de seleção do BERTimbau (melhor nos mesmos folds)
            m['cv_selecao_mesmos_folds'] = _margem(bert_cv, bert_por_fold, b, 'cv_plana_melhor')
        dp_s = teste_bert['f1_macro_dp_entre_sementes']
        margem_t = teste_bert['f1_macro_media'] - b['teste_f1_macro']
        m['teste'] = {
            'teste_bertimbau_media_sementes': teste_bert['f1_macro_media'],
            'teste_bertimbau_semente_referencia': teste_bert['f1_macro_semente_referencia'],
            'teste_baseline': b['teste_f1_macro'],
            'margem_media_sementes': float(margem_t),
            'margem_semente_referencia': float(teste_bert['f1_macro_semente_referencia']
                                               - b['teste_f1_macro']),
            'margem_em_dp_sementes': float(margem_t / dp_s) if dp_s else None,
        }
        por_baseline[nome] = m
    mais_forte = max(baselines, key=lambda n: baselines[n]['cv']['media'])
    mais_forte_teste = max(baselines, key=lambda n: baselines[n]['teste_f1_macro'])
    return {
        'baseline_mais_forte_cv': mais_forte,
        'baseline_mais_forte_teste': mais_forte_teste,
        'contra_mais_forte_cv': por_baseline[mais_forte],
        'por_baseline': por_baseline,
    }


# ----------------------------------------------------------------------------
# Tabela 3
# ----------------------------------------------------------------------------

CABECALHO_TEX = (
    '% Gerado por app/nlp/evaluation/scripts/analisar_dispersao.py\n'
    '% NAO EDITAR A MAO: reexecute o script apos retreinar.\n'
)


def _n(valor, casas=4):
    return f'{valor:.{casas}f}'


def escrever_tab_baselines(resumo, destino_dir, duas_colunas=True):
    """
    Tabela 3: F1 de CV (média ± DP entre os mesmos folds) e F1 no teste retido.

    BERTimbau no teste: média ± DP entre sementes, com o modelo de referência no
    texto da legenda. Baselines ajustadas (CV aninhada) e fixas em blocos próprios.
    """
    bert, teste = resumo['bertimbau'], resumo['bertimbau']['teste']
    baselines = resumo['baselines']
    cv = bert['cv']
    ajustadas = sorted([n for n, b in baselines.items() if b['tipo'] == 'ajustada'],
                       key=lambda n: -baselines[n]['cv']['media'])
    fixas = sorted([n for n, b in baselines.items() if b['tipo'] == 'fixa'],
                   key=lambda n: -baselines[n]['cv']['media'])
    n_cfg = bert.get('n_configuracoes_busca')
    n_cfg_aj = sorted({baselines[n]['n_configuracoes'] for n in ajustadas})

    linhas = []
    add = linhas.append
    add('\\begin{table*}[!t]' if duas_colunas else '\\begin{table}[!t]')
    add('\\centering')
    partes = [
        'Macro F1 of the fine-tuned BERTimbau and of the TF-IDF baselines. '
        'Cross-validation (CV) uses the same five video-grouped folds for every model, '
        'and its dispersion is always the standard deviation across these folds.',
    ]
    if cv.get('n_sementes', 1) > 1:
        partes.append(
            f"BERTimbau's CV score is that of the configuration selected among {n_cfg} on "
            f"these folds, averaged over {cv['n_sementes']} seeds (fold SD averaged over seeds; "
            f"the SD across seeds is {_n(cv['dp_entre_sementes'], 4)}).")
    else:
        partes.append(f"BERTimbau's CV score is that of the configuration selected among "
                      f"{n_cfg} on these folds.")
    if ajustadas:
        partes.append(
            f"Tuned baselines were searched over {'/'.join(map(str, n_cfg_aj))} configurations "
            "each (word and character $n$-gram TF-IDF) and are reported with nested CV, "
            "free of selection optimism.")
    partes.append(
        f"Test: held-out set of {teste['dados_treino_final']['n_teste']} sentences; BERTimbau "
        f"is the mean $\\pm$ SD over {teste['n_sementes']} training seeds (the reference model, "
        f"seed {teste['semente_referencia']}, used in the remaining analyses, reaches "
        f"{_n(teste['f1_macro_semente_referencia'])}); each baseline is a single deterministic fit.")
    add('\\caption{' + ' '.join(partes) + '}')
    add('\\label{tab:baselines}')
    add('\\begin{tabular}{llcc}')
    add('\\toprule')
    add('Model & Selection & CV macro F1 & Test macro F1 \\\\')
    add('\\midrule')
    celulas = [NOME_BERT, f'best of {n_cfg}',
               f"{_n(cv['media'])} $\\pm$ {_n(cv['dp_entre_folds'], 3)}",
               f"{_n(teste['f1_macro_media'])} $\\pm$ {_n(teste['f1_macro_dp_entre_sementes'], 3)}"]
    add(' & '.join(f'\\textbf{{{c}}}' for c in celulas) + ' \\\\')
    for bloco, nomes in (('ajustadas', ajustadas), ('fixas', fixas)):
        if not nomes:
            continue
        add('\\midrule')
        for nome in nomes:
            b = baselines[nome]
            selecao = (f"nested CV, {b['n_configuracoes']}" if b['tipo'] == 'ajustada'
                       else 'fixed')
            add(' & '.join([
                nome, selecao,
                f"{_n(b['cv']['media'])} $\\pm$ {_n(b['cv']['dp_entre_folds'], 3)}",
                _n(b['teste_f1_macro']),
            ]) + ' \\\\')
    add('\\bottomrule')
    add('\\end{tabular}')
    add('\\end{table*}' if duas_colunas else '\\end{table}')

    caminho = os.path.join(str(destino_dir), 'tab-baselines.tex')
    with open(caminho, 'w', encoding='utf-8') as f:
        f.write(CABECALHO_TEX + '\n'.join(linhas) + '\n')
    return caminho


# ----------------------------------------------------------------------------

def resumir(pasta):
    """Lê as entradas de `pasta` e devolve o resumo de dispersão (sem gravar)."""
    grid, arq_grid = _carregar(pasta, os.path.join('toxicity_gridsearch',
                                                   'gridsearch_optimized_search_*.json'))
    final, arq_final = _carregar(pasta, os.path.join('modelo_final', 'modelo_final_*.json'))
    base, arq_base = _carregar(pasta, 'baselines_toxicity_*.json', obrigatorio=False)
    ajust, arq_ajust = _carregar(pasta, 'baselines_ajustados_*.json', obrigatorio=False)
    if base is None and ajust is None:
        raise FileNotFoundError(f"Nenhum JSON de baselines (fixas ou ajustadas) em {pasta}")

    bert = dispersao_bertimbau(grid)
    bert['teste'] = teste_bertimbau(final, os.path.join(pasta, 'logs', 'final.log'))
    if final.get('config') and final['config'] != bert['config']:
        raise ValueError(f"modelo_final ({arq_final}) usa outra configuração que a busca "
                         f"({arq_grid}); confira as rodadas")

    baselines = {}
    if ajust:
        baselines.update(baselines_ajustadas(ajust))
    if base:
        baselines.update(baselines_fixas(base))

    return {
        'experimento': (final.get('experimento') or grid.get('experimento')),
        'fontes': {'pasta': os.path.abspath(pasta), 'gridsearch': arq_grid,
                   'modelo_final': arq_final, 'baselines_fixas': arq_base,
                   'baselines_ajustadas': arq_ajust},
        'definicoes': {
            'dp_entre_folds': 'desvio-padrão amostral (ddof=1) do F1 macro entre os 5 folds '
                              'gravados, agrupados por vídeo, os mesmos para todos os modelos',
            'dp_entre_sementes': 'desvio-padrão amostral (ddof=1) entre as médias de CV de '
                                 'cada semente (BERTimbau, estágio 3); grandeza distinta, '
                                 'reportada à parte',
            'teste_bertimbau': 'média ± DP amostral entre as sementes do treino final',
            'margem_em_dp_folds_bertimbau': 'margem de CV dividida pelo DP entre folds do '
                                            'BERTimbau (o cálculo do revisor: 5,3 pontos ~ 1,9x)',
            'pareado_por_fold': 'diferença de F1 fold a fold entre BERTimbau (estágio 2, '
                                'semente 42) e a baseline; descritivo, sem teste: com 5 folds '
                                'dependentes, um teste t ingênuo seria anticonservador',
            'cv_baselines_ajustadas': 'CV aninhada (seleção nos folds internos); '
                                      'cv_selecao_mesmos_folds usa a melhor configuração '
                                      'escolhida nos mesmos 5 folds, a regra do BERTimbau',
        },
        'bertimbau': bert,
        'baselines': baselines,
        'margens': calcular_margens(bert, bert['teste'], baselines),
    }


def main():
    ap = argparse.ArgumentParser(description='Dispersão comparável (M3a), teste por sementes '
                                             '(M3c) e Tabela 3')
    ap.add_argument('--resultados', default=None,
                    help='pasta de resultados de onde ler as entradas '
                         '(padrão: experimento.dir_resultados())')
    ap.add_argument('--saida', default=None,
                    help='pasta para o JSON e a tabela (padrão: JSON em dir_resultados() e '
                         "tabela em experimento.dir_latex('tabelas'))")
    ap.add_argument('--uma-coluna', action='store_true', help='table em vez de table*')
    args = ap.parse_args()

    pasta = os.path.abspath(args.resultados) if args.resultados else str(experimento.dir_resultados())
    if args.saida:
        destino_json = destino_tex = os.path.abspath(args.saida)
        os.makedirs(destino_json, exist_ok=True)
    else:
        if args.resultados and os.path.abspath(args.resultados) != str(experimento.dir_resultados()):
            raise SystemExit("--resultados aponta para outra rodada: use também --saida, para "
                             "não gravar a tabela de outra rodada na pasta desta.")
        destino_json = str(experimento.dir_resultados())
        destino_tex = str(experimento.dir_latex('tabelas'))

    resumo = resumir(pasta)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    resumo['timestamp'] = ts
    arq_json = os.path.join(destino_json, f'dispersao_{ts}.json')
    with open(arq_json, 'w', encoding='utf-8') as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False, default=str)
    arq_tex = escrever_tab_baselines(resumo, destino_tex, duas_colunas=not args.uma_coluna)

    bert, teste = resumo['bertimbau'], resumo['bertimbau']['teste']
    cv = bert['cv']
    print('=' * 84)
    print('DISPERSÃO COMPARÁVEL — F1 MACRO (DP entre os mesmos 5 folds, ddof=1)')
    print('=' * 84)
    print(f"{'Modelo':<34} {'CV (DP folds)':>20} {'Teste':>20}")
    print('-' * 84)
    print(f"{NOME_BERT:<34} {cv['media']:>11.4f} ± {cv['dp_entre_folds']:.4f} "
          f"{teste['f1_macro_media']:>11.4f} ± {teste['f1_macro_dp_entre_sementes']:.4f}")
    for nome, b in sorted(resumo['baselines'].items(), key=lambda kv: -kv[1]['cv']['media']):
        print(f"{nome:<34} {b['cv']['media']:>11.4f} ± {b['cv']['dp_entre_folds']:.4f} "
              f"{b['teste_f1_macro']:>20.4f}")
    print('-' * 84)
    if cv.get('dp_entre_sementes') is not None:
        print(f"BERTimbau, DP entre sementes (CV): {cv['dp_entre_sementes']:.4f} "
              f"({cv['n_sementes']} sementes)")
    print(f"BERTimbau, teste da semente de referência ({teste['semente_referencia']}): "
          f"{teste['f1_macro_semente_referencia']:.4f}")
    m = resumo['margens']
    mf = m['contra_mais_forte_cv']['cv']
    print(f"Margem de CV sobre a baseline mais forte ({m['baseline_mais_forte_cv']}): "
          f"{mf['margem']:+.4f} = {mf['margem_em_dp_folds_bertimbau']:.2f}x o DP entre folds "
          f"do BERTimbau")
    if mf.get('pareado_por_fold'):
        p = mf['pareado_por_fold']
        print(f"  fold a fold: média {p['media']:+.4f}, DP {p['dp']:.4f}; BERTimbau melhor em "
              f"{p['folds_bertimbau_melhor']}/{p['n_folds']} folds")
    print('=' * 84)
    print(f"\nSalvo em: {arq_json}\nTabela:   {arq_tex}")


if __name__ == '__main__':
    main()
