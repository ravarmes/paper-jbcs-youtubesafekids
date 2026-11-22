"""
Baselines clássicas AJUSTADAS, com validação cruzada aninhada.

Responde ao item M3b da revisão do JBCS (otimismo de seleção). O F1 de validação
cruzada do BERTimbau é o da melhor de 32 configurações, escolhida nos mesmos
folds em que é reportado; as baselines de train_toxicity_baselines.py são fixas
(TF-IDF de palavras 1-2, 10.000 termos). Aqui as baselines lineares recebem um
orçamento de busca comparável -- 32 configurações sorteadas por modelo, como na
busca aleatória do BERTimbau -- incluindo TF-IDF de n-gramas de CARACTERES
(char_wb, 2 a 5), forte em texto curto e ruidoso.

Protocolo (partição por vídeo, YSK_PARTICAO='video'):

  CV aninhada -- estimativa sem otimismo de seleção
    folds externos: os 5 folds gravados em data/particoes/particao_video.csv
                    (fold_k5), os mesmos da busca do BERTimbau;
    folds internos: GroupKFold (grupo = vídeo) dentro do treino de cada fold
                    externo. Cada configuração é avaliada nos folds internos; a
                    de maior F1 macro médio é retreinada no treino externo
                    inteiro e avaliada no fold externo, que não participou da
                    escolha. Reporta-se média e desvio-padrão ENTRE OS 5 FOLDS
                    externos e a configuração escolhida em cada um.

  CV plana -- o análogo do número do BERTimbau
    todas as configurações avaliadas nos 5 folds gravados; a melhor é escolhida
    e reportada nos mesmos folds (mesmo critério da busca do BERTimbau, logo com
    o mesmo otimismo de seleção). A diferença entre CV plana e aninhada mede
    esse otimismo para as baselines.

  Modelo final
    a configuração vencedora da CV plana (seleção por CV agrupada no treino
    inteiro, com os folds gravados) é retreinada em todo o treino e avaliada uma
    única vez no teste retido. As predições do teste vão para um CSV com
    id_sentenca, lido por comparar_modelos.py para o bootstrap pareado.

Saídas, em experimento.dir_resultados() (ou --saida):
    baselines_ajustados_<timestamp>.json
    predicoes_teste_baselines_ajustados_<timestamp>.csv

Uso (de dentro de src/, só CPU; o script não importa o torch):
    python app/nlp/training/train_toxicity_baselines_ajustados.py
    python app/nlp/training/train_toxicity_baselines_ajustados.py --n-jobs 4 --saida <pasta>
    (--n-jobs > 1 abre processos; com um treino em GPU na mesma máquina, deixe 1)
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp import experimento  # noqa: E402
from app.nlp.datasets.prepare_data_toxicity import get_data_for_cv_and_test  # noqa: E402


def _carregar_get_kfold_split():
    """
    get_kfold_split de app/nlp/utils/data_utils.py, carregado pelo caminho do arquivo.

    `from app.nlp.utils.data_utils import ...` executaria app/nlp/utils/__init__.py,
    que importa training_utils e, com ele, o torch (~2 GB de memória comprometida
    no Windows, só para gerar índices de folds). Carregar o módulo pelo caminho
    usa a mesma função sem o pacote; data_utils só depende de pandas/numpy/sklearn.
    """
    import importlib.util
    caminho = os.path.join(project_root, 'app', 'nlp', 'utils', 'data_utils.py')
    spec = importlib.util.spec_from_file_location('_ysk_data_utils', caminho)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo.get_kfold_split


get_kfold_split = _carregar_get_kfold_split()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ======================================================================
# Configuração
# ======================================================================

K_FOLDS = 5              # folds externos: os gravados na partição por vídeo
K_INTERNO = 4            # GroupKFold interno, por vídeo, no treino de cada fold externo
N_CONFIGS = 32           # mesmo número de configurações da busca do BERTimbau
RANDOM_STATE = 42
N_JOBS_PADRAO = 1        # serial por padrão: processos extras disputam memória com
                         # um treino em GPU que esteja em curso na mesma máquina
CLASS_NAMES = ['Nenhuma', 'Leve', 'Severa']

# Representações TF-IDF. Cada vetorizador normaliza o próprio bloco (L2); na
# união palavra+caractere os dois blocos são concatenados.
FEATURES = {
    'word_1_1': [('word', (1, 1))],
    'word_1_2': [('word', (1, 2))],
    'word_1_3': [('word', (1, 3))],
    'char_wb_2_5': [('char_wb', (2, 5))],
    'word_1_2+char_wb_2_5': [('word', (1, 2)), ('char_wb', (2, 5))],
}

_ESPACO_TFIDF = {
    'features': list(FEATURES),
    'max_features': [10000, 50000, None],
    'sublinear_tf': [True, False],
}

# Espaço de busca por modelo. O sorteio cobre cada dimensão isolada melhor do que
# uma grade grosseira de mesmo custo (Bergstra & Bengio, 2012) -- o mesmo
# argumento da busca do BERTimbau.
ESPACOS = {
    'svm': dict(_ESPACO_TFIDF, C=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                class_weight=[None, 'balanced']),
    'lr': dict(_ESPACO_TFIDF, C=[0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
               class_weight=[None, 'balanced']),
    'cnb': dict(_ESPACO_TFIDF, alpha=[0.01, 0.03, 0.1, 0.3, 1.0]),
}

NOMES = {
    'svm': 'SVM (LinearSVC), tuned',
    'lr': 'Logistic Regression, tuned',
    'cnb': 'Complement NB, tuned',
}


def amostrar_configuracoes(espaco, n_amostras, random_state=RANDOM_STATE):
    """
    Sorteia `n_amostras` configurações distintas do espaço de busca.

    Mesmo algoritmo de amostrar_configuracoes() em train_toxicity_gridsearch.py
    (reimplementado aqui para não importar o torch num script só de CPU).
    """
    rng = np.random.RandomState(random_state)
    nomes = list(espaco.keys())
    vistas, configs = set(), []
    total_possivel = int(np.prod([len(v) for v in espaco.values()]))
    alvo = min(n_amostras, total_possivel)
    while len(configs) < alvo:
        escolha = tuple(valores[rng.randint(len(valores))] for valores in espaco.values())
        if escolha in vistas:
            continue
        vistas.add(escolha)
        configs.append(dict(zip(nomes, escolha)))
    return configs


def montar_pipeline(modelo, cfg):
    """TF-IDF (palavra, caractere ou união) + classificador, conforme a configuração."""
    vetorizadores = [
        (analyzer, TfidfVectorizer(analyzer=analyzer, ngram_range=ngram,
                                   max_features=cfg['max_features'],
                                   sublinear_tf=cfg['sublinear_tf']))
        for analyzer, ngram in FEATURES[cfg['features']]
    ]
    vet = vetorizadores[0][1] if len(vetorizadores) == 1 else FeatureUnion(vetorizadores)

    if modelo == 'svm':
        # dual=True é o padrão do scikit-learn 1.3; explícito só para calar o
        # FutureWarning repetido a cada ajuste (o padrão muda para 'auto' na 1.5).
        clf = LinearSVC(C=cfg['C'], class_weight=cfg['class_weight'], dual=True,
                        random_state=RANDOM_STATE, max_iter=10000)
    elif modelo == 'lr':
        clf = LogisticRegression(C=cfg['C'], class_weight=cfg['class_weight'],
                                 random_state=RANDOM_STATE, max_iter=3000,
                                 solver='lbfgs', multi_class='multinomial')
    elif modelo == 'cnb':
        clf = ComplementNB(alpha=cfg['alpha'])
    else:
        raise ValueError(f"modelo desconhecido: {modelo}")
    return Pipeline([('tfidf', vet), ('clf', clf)])


def _ajustar_e_pontuar(modelo, cfg, X, y, idx_tr, idx_va):
    """Treina no índice de treino e devolve o F1 macro no de validação."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        pipe = montar_pipeline(modelo, cfg).fit(X[idx_tr], y[idx_tr])
    pred = pipe.predict(X[idx_va])
    return float(f1_score(y[idx_va], pred, average='macro', zero_division=0))


def _metricas(y_true, y_pred):
    return {
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
    }


def _resumo_folds(valores):
    """Média e desvio-padrão entre folds; ddof=1 é o que se compara entre modelos."""
    v = np.asarray(valores, dtype=float)
    return {
        'media': float(v.mean()),
        'dp_entre_folds': float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        'dp_entre_folds_ddof0': float(v.std(ddof=0)),
        'por_fold': v.tolist(),
    }


# ======================================================================
# CV plana e aninhada
# ======================================================================

def avaliar_grade(modelo, configs, X, y, divisoes, n_jobs):
    """
    F1 de cada configuração em cada divisão (treino, validação).

    Returns:
        matriz [n_configs, n_divisoes] de F1 macro.
    """
    tarefas = [(i, j) for i in range(len(configs)) for j in range(len(divisoes))]
    valores = Parallel(n_jobs=n_jobs)(
        delayed(_ajustar_e_pontuar)(modelo, configs[i], X, y, *divisoes[j])
        for i, j in tarefas
    )
    matriz = np.zeros((len(configs), len(divisoes)))
    for (i, j), v in zip(tarefas, valores):
        matriz[i, j] = v
    return matriz


def cv_aninhada(modelo, configs, X, y, g, folds_externos, n_jobs):
    """Seleção por GroupKFold interno (vídeo) e avaliação no fold externo."""
    por_fold, f1s, accs = [], [], []
    for k, (tr, va) in enumerate(folds_externos, 1):
        internos = [(tr[a], tr[b]) for a, b in
                    GroupKFold(n_splits=K_INTERNO).split(X[tr], y[tr], groups=g[tr])]
        matriz = avaliar_grade(modelo, configs, X, y, internos, n_jobs)
        medias = matriz.mean(axis=1)
        escolhida = int(np.argmax(medias))       # empate: a primeira sorteada
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            pipe = montar_pipeline(modelo, configs[escolhida]).fit(X[tr], y[tr])
        pred = pipe.predict(X[va])
        m = _metricas(y[va], pred)
        f1s.append(m['f1_macro'])
        accs.append(m['accuracy'])
        por_fold.append({
            'fold': k,
            'n_treino': int(len(tr)), 'n_validacao': int(len(va)),
            'n_videos_treino': int(len(set(g[tr]))), 'n_videos_validacao': int(len(set(g[va]))),
            'config_escolhida_idx': escolhida,
            'config_escolhida': configs[escolhida],
            'f1_interno_medio_escolhida': float(medias[escolhida]),
            'f1_interno_por_fold_escolhida': matriz[escolhida].tolist(),
            **{f'externo_{c}': v for c, v in m.items()},
        })
        logger.info(f"  [{modelo}] fold externo {k}: config #{escolhida} "
                    f"(F1 interno {medias[escolhida]:.4f}) -> F1 externo {m['f1_macro']:.4f}")
    return {
        'f1_macro': _resumo_folds(f1s),
        'accuracy': _resumo_folds(accs),
        'por_fold': por_fold,
        'n_configs_distintas_escolhidas': len({f['config_escolhida_idx'] for f in por_fold}),
    }


def cv_plana(modelo, configs, X, y, folds, n_jobs):
    """Todas as configurações nos 5 folds gravados; a melhor é a do modelo final."""
    matriz = avaliar_grade(modelo, configs, X, y, folds, n_jobs)
    medias = matriz.mean(axis=1)
    melhor = int(np.argmax(medias))
    ranking = np.argsort(-medias)
    return {
        'melhor_idx': melhor,
        'melhor_config': configs[melhor],
        'melhor_f1_macro': _resumo_folds(matriz[melhor]),
        'todas': [{'idx': int(i), 'config': configs[i], **_resumo_folds(matriz[i])}
                  for i in ranking],
    }


# ======================================================================
# Teste retido
# ======================================================================

def avaliar_teste(modelo, cfg, X_tr, y_tr, X_te, y_te):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        pipe = montar_pipeline(modelo, cfg).fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    tfidf = pipe.named_steps['tfidf']
    n_termos = (sum(len(v.vocabulary_) for _, v in tfidf.transformer_list)
                if isinstance(tfidf, FeatureUnion) else len(tfidf.vocabulary_))
    res = {
        **_metricas(y_te, pred),
        'f1_per_class': f1_score(y_te, pred, average=None, labels=[0, 1, 2], zero_division=0).tolist(),
        'precision_per_class': precision_score(y_te, pred, average=None, labels=[0, 1, 2],
                                               zero_division=0).tolist(),
        'recall_per_class': recall_score(y_te, pred, average=None, labels=[0, 1, 2],
                                         zero_division=0).tolist(),
        'support_per_class': np.bincount(y_te, minlength=3).tolist(),
        'confusion_matrix': confusion_matrix(y_te, pred, labels=[0, 1, 2]).tolist(),
        'classification_report': classification_report(y_te, pred, labels=[0, 1, 2],
                                                       target_names=CLASS_NAMES,
                                                       zero_division=0),
        'n_termos_vocabulario': int(n_termos),
    }
    return res, pred


def _por_similaridade(X_tr, X_te, y_te, preds):
    """Mesma separação com/sem análogo próximo usada nas demais avaliações."""
    try:
        from app.nlp.evaluation.evaluate_toxicity import (
            separar_por_similaridade, LIMIAR_SIMILARIDADE,
        )
    except Exception as e:  # noqa: BLE001 -- análise acessória, não bloqueia
        logger.warning(f"Separação por similaridade indisponível: {e}")
        return None, {}
    com, sem, _ = separar_por_similaridade(list(X_tr), list(X_te), limiar=LIMIAR_SIMILARIDADE)
    saida = {}
    for chave, pred in preds.items():
        bloco = {'limiar': LIMIAR_SIMILARIDADE}
        for nome, idx in (('com_analogo', com), ('sem_analogo', sem)):
            bloco[nome] = ({'n': len(idx),
                            'f1': float(f1_score(y_te[idx], pred[idx], average='macro',
                                                 zero_division=0)),
                            'accuracy': float(accuracy_score(y_te[idx], pred[idx]))}
                           if idx else None)
        saida[chave] = bloco
    return {'limiar': LIMIAR_SIMILARIDADE, 'n_com_analogo': len(com),
            'n_sem_analogo': len(sem)}, saida


# ======================================================================

def main():
    ap = argparse.ArgumentParser(description='Baselines TF-IDF ajustadas com CV aninhada (M3b)')
    ap.add_argument('--modelos', nargs='+', choices=list(ESPACOS), default=list(ESPACOS))
    ap.add_argument('--n-configs', type=int, default=N_CONFIGS)
    ap.add_argument('--n-jobs', type=int, default=N_JOBS_PADRAO)
    ap.add_argument('--saida', default=None,
                    help='pasta de saída (padrão: experimento.dir_resultados())')
    ap.add_argument('--similaridade', action='store_true',
                    help='separa o teste em frases com/sem análogo próximo no treino '
                         '(importa evaluate_toxicity, e com ele o torch)')
    args = ap.parse_args()

    t0 = time.time()
    saida = os.path.abspath(args.saida) if args.saida else str(experimento.dir_resultados())
    os.makedirs(saida, exist_ok=True)

    (X_l, y_l, g_l), (Xt_l, yt_l), (df, train_idx, test_idx) = get_data_for_cv_and_test(
        test_size=0.20, return_groups=True, return_frame=True)
    X, y, g = np.array(X_l, dtype=object), np.array(y_l), np.array(g_l).astype(str)
    X_te, y_te = np.array(Xt_l, dtype=object), np.array(yt_l)
    ids_teste = df['id_sentenca'].values[test_idx]
    videos_teste = (df['ID'].astype(str).values[test_idx] if 'ID' in df.columns
                    else np.array([''] * len(test_idx)))

    folds = [(np.asarray(tr), np.asarray(va)) for tr, va in
             get_kfold_split(X, y, n_splits=K_FOLDS, random_state=RANDOM_STATE, groups=g)]
    logger.info(f"Desenvolvimento: {len(X)} frases, {len(set(g))} grupos; teste: {len(X_te)} frases")
    logger.info("Folds externos: " + ", ".join(f"{k}: {len(va)} frases"
                                                for k, (_, va) in enumerate(folds, 1)))

    resultados, preds_teste = {}, {}
    for modelo in args.modelos:
        configs = amostrar_configuracoes(ESPACOS[modelo], args.n_configs)
        logger.info(f"\n{'=' * 70}\n{NOMES[modelo]}: {len(configs)} configurações\n{'=' * 70}")
        t_mod = time.time()

        aninhada = cv_aninhada(modelo, configs, X, y, g, folds, args.n_jobs)
        plana = cv_plana(modelo, configs, X, y, folds, args.n_jobs)
        teste, pred = avaliar_teste(modelo, plana['melhor_config'], X, y, X_te, y_te)
        preds_teste[modelo] = pred

        resultados[modelo] = {
            'nome': NOMES[modelo],
            'espaco_busca': ESPACOS[modelo],
            'n_configuracoes': len(configs),
            'configuracoes': configs,
            'cv_aninhada': aninhada,
            'cv_plana': plana,
            'teste': {'config': plana['melhor_config'], **teste},
            'n_ajustes': len(folds) * len(configs) * K_INTERNO + len(folds)
                         + len(folds) * len(configs) + 1,
            'segundos': round(time.time() - t_mod, 1),
        }
        an, pl = aninhada['f1_macro'], plana['melhor_f1_macro']
        logger.info(f"  CV aninhada: {an['media']:.4f} ± {an['dp_entre_folds']:.4f} (DP entre folds)")
        logger.info(f"  CV plana (melhor config #{plana['melhor_idx']}): "
                    f"{pl['media']:.4f} ± {pl['dp_entre_folds']:.4f}")
        logger.info(f"  Teste: F1 macro {teste['f1_macro']:.4f} | acc {teste['accuracy']:.4f}")

    sim_info = None
    if args.similaridade:
        sim_info, sim = _por_similaridade(X, X_te, y_te, preds_teste)
        for modelo, bloco in sim.items():
            resultados[modelo]['teste']['por_similaridade'] = bloco

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    arq_pred = f'predicoes_teste_baselines_ajustados_{ts}.csv'
    tabela = pd.DataFrame({'id_sentenca': ids_teste, 'video_id': videos_teste, 'rotulo': y_te})
    for modelo, pred in preds_teste.items():
        tabela[f'pred_{modelo}'] = pred
    tabela.to_csv(os.path.join(saida, arq_pred), sep=';', index=False, encoding='utf-8')

    saida_json = {
        'timestamp': ts,
        'experimento': experimento.descricao(),
        'metodologia': {
            'objetivo': 'M3b: baselines ajustadas com orçamento comparável ao do BERTimbau '
                        '(32 configurações), com CV aninhada para estimativa sem otimismo de seleção',
            'folds_externos': (f'{K_FOLDS} folds gravados, agrupados por vídeo '
                               '(data/particoes/particao_video.csv, coluna fold_k5)'
                               if experimento.particao() == 'video' else
                               f'{K_FOLDS} folds de get_kfold_split (partição por frase)'),
            'folds_internos': f'GroupKFold(n_splits={K_INTERNO}), grupo = vídeo, '
                              'dentro do treino de cada fold externo',
            'criterio_selecao': 'maior F1 macro médio nos folds internos (empate: primeira sorteada)',
            'cv_plana': 'todas as configurações nos 5 folds externos; melhor reportada nos mesmos '
                        'folds (mesmo otimismo de seleção do número de CV do BERTimbau)',
            'modelo_final': 'melhor configuração da CV plana, retreinada no treino inteiro e '
                            'avaliada uma vez no teste retido',
            'dispersao': "dp_entre_folds usa ddof=1 (desvio amostral entre os 5 folds); "
                         "dp_entre_folds_ddof0 é o np.std padrão, para conferência",
            'amostragem_configs': f'busca aleatória sem reposição, random_state={RANDOM_STATE}',
            'features': {k: [{'analyzer': a, 'ngram_range': list(n)} for a, n in v]
                         for k, v in FEATURES.items()},
            'n_configs': args.n_configs,
            'k_interno': K_INTERNO,
            'random_state': RANDOM_STATE,
        },
        'split': {'n_dev': int(len(X)), 'n_teste': int(len(X_te)),
                  'n_grupos_dev': int(len(set(g))),
                  'n_por_fold_externo': [int(len(va)) for _, va in folds]},
        'similaridade_teste': sim_info,
        'arquivo_predicoes_teste': arq_pred,
        'resultados': resultados,
        'segundos_total': round(time.time() - t0, 1),
    }
    arq_json = os.path.join(saida, f'baselines_ajustados_{ts}.json')
    with open(arq_json, 'w', encoding='utf-8') as f:
        json.dump(saida_json, f, indent=2, ensure_ascii=False, default=str)

    print('\n' + '=' * 92)
    print('BASELINES AJUSTADAS — F1 MACRO')
    print('=' * 92)
    print(f"{'Modelo':<30} {'CV aninhada (DP folds)':>24} {'CV plana, melhor':>20} {'Teste':>9}")
    print('-' * 92)
    for modelo, r in resultados.items():
        an, pl = r['cv_aninhada']['f1_macro'], r['cv_plana']['melhor_f1_macro']
        print(f"{r['nome']:<30} {an['media']:>14.4f} ± {an['dp_entre_folds']:.4f} "
              f"{pl['media']:>11.4f} ± {pl['dp_entre_folds']:.4f} {r['teste']['f1_macro']:>9.4f}")
        print(f"{'':<30} config final: {r['teste']['config']}")
    print('=' * 92)
    print(f"\nSalvo em: {arq_json}\nPredições do teste: {os.path.join(saida, arq_pred)}")
    print(f"Tempo total: {(time.time() - t0) / 60:.1f} min")


if __name__ == '__main__':
    main()
