"""
Baselines Clássicas para Detecção de Toxicidade.

Compara TF-IDF + modelos tradicionais com o BERTimbau fine-tuned.
Usa a mesma divisão de dados e metodologia (cross-validation 5 folds)
do grid search principal para garantir comparabilidade.

Modelos:
    1. SVM (LinearSVC)
    2. Logistic Regression
    3. Random Forest
    4. Naive Bayes (MultinomialNB)
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline

# Setup do path do projeto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp import experimento  # noqa: E402

from app.nlp.datasets.prepare_data_toxicity import get_data_for_cv_and_test
from app.nlp.utils.data_utils import get_kfold_split
from app.nlp.config import PATHS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ======================================================================
# Configuração
# ======================================================================

K_FOLDS = 5
RANDOM_STATE = 42
CLASS_NAMES = ['Nenhuma', 'Leve', 'Severa']

# Definição dos modelos baseline
BASELINES = {
    'SVM (LinearSVC)': Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ('clf', LinearSVC(
            class_weight='balanced',
            random_state=RANDOM_STATE,
            max_iter=10000
        ))
    ]),
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(
            class_weight='balanced',
            random_state=RANDOM_STATE,
            max_iter=1000,
            solver='lbfgs',
            multi_class='multinomial'
        ))
    ]),
    'Random Forest': Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]),
    'Naive Bayes': Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ('clf', MultinomialNB(alpha=1.0))
    ])
}


def evaluate_with_cv(X_train_cv, y_train_cv, groups=None):
    """
    Avalia todas as baselines com cross-validation estratificada (K folds).

    Usa exatamente os mesmos folds do grid search do BERTimbau (mesma semente,
    mesmos grupos), para que a comparação entre as duas famílias de modelos
    seja feita sobre as mesmas partições.

    Args:
        X_train_cv: Lista de textos de treino (para CV)
        y_train_cv: Lista de labels de treino
        groups: Grupos (texto normalizado) para folds sem vazamento de frases

    Returns:
        Dict com resultados por modelo
    """
    X = np.array(X_train_cv)
    y = np.array(y_train_cv)
    g = np.array(groups) if groups is not None else None

    results = {}

    for name, pipeline in BASELINES.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Avaliando: {name}")
        logger.info(f"{'='*60}")

        fold_metrics = {
            'f1': [], 'accuracy': [], 'precision': [], 'recall': []
        }

        for fold, (train_idx, val_idx) in enumerate(
            get_kfold_split(X, y, n_splits=K_FOLDS, random_state=RANDOM_STATE, groups=g), 1
        ):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # Treina o pipeline (TF-IDF + Classificador)
            from sklearn.base import clone
            model = clone(pipeline)
            model.fit(X_fold_train, y_fold_train)
            
            # Predição
            y_pred = model.predict(X_fold_val)
            
            # Métricas
            f1 = f1_score(y_fold_val, y_pred, average='macro')
            acc = accuracy_score(y_fold_val, y_pred)
            prec = precision_score(y_fold_val, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_fold_val, y_pred, average='macro', zero_division=0)
            
            fold_metrics['f1'].append(f1)
            fold_metrics['accuracy'].append(acc)
            fold_metrics['precision'].append(prec)
            fold_metrics['recall'].append(rec)
            
            logger.info(f"  Fold {fold}/{K_FOLDS}: F1={f1:.4f} | Acc={acc:.4f}")
        
        # Média e desvio padrão. 'f1_std' é o np.std padrão (ddof=0), mantido
        # por compatibilidade; 'f1_dp_entre_folds' (ddof=1) é o mesmo tipo de
        # dispersão que a busca do BERTimbau grava (pandas .std()), e é o que se
        # compara entre modelos (M3a). Os F1 por fold ficam no JSON para as
        # diferenças pareadas por fold (analisar_dispersao.py).
        results[name] = {
            'f1_mean': np.mean(fold_metrics['f1']),
            'f1_std': np.std(fold_metrics['f1']),
            'f1_dp_entre_folds': float(np.std(fold_metrics['f1'], ddof=1)),
            'f1_por_fold': [float(v) for v in fold_metrics['f1']],
            'accuracy_mean': np.mean(fold_metrics['accuracy']),
            'accuracy_std': np.std(fold_metrics['accuracy']),
            'precision_mean': np.mean(fold_metrics['precision']),
            'precision_std': np.std(fold_metrics['precision']),
            'recall_mean': np.mean(fold_metrics['recall']),
            'recall_std': np.std(fold_metrics['recall']),
            'fold_details': fold_metrics
        }
        
        logger.info(f"  MÉDIA: F1={results[name]['f1_mean']:.4f} (±{results[name]['f1_std']:.4f})")
    
    return results


def deduplicar(textos, rotulos):
    """
    Mantém a primeira ocorrência de cada texto distinto.

    O corpus repete frases-molde; na métrica por ocorrência uma frase repetida
    30 vezes pesa 30 vezes. Avaliar também sobre textos distintos separa
    desempenho de frequência.
    """
    vistos = {}
    for texto, rotulo in zip(textos, rotulos):
        chave = str(texto).strip().lower()
        vistos.setdefault(chave, (texto, rotulo))
    return ([t for t, _ in vistos.values()], [r for _, r in vistos.values()])


def evaluate_on_test(X_train_cv, y_train_cv, X_test, y_test):
    """
    Treina cada baseline no conjunto de treino completo e avalia no teste.
    
    Args:
        X_train_cv: Textos de treino
        y_train_cv: Labels de treino
        X_test: Textos de teste (600 amostras reservadas)
        y_test: Labels de teste
        
    Returns:
        Dict com resultados por modelo no teste
    """
    X_train = np.array(X_train_cv)
    y_train = np.array(y_train_cv)
    X_tst = np.array(X_test)
    y_tst = np.array(y_test)
    
    X_dist, y_dist = deduplicar(X_test, y_test)
    X_dist, y_dist = np.array(X_dist), np.array(y_dist)

    # Mesma partição por proximidade usada na avaliação do BERTimbau, para que
    # os dois modelos sejam comparados nos mesmos subconjuntos.
    from app.nlp.evaluation.evaluate_toxicity import (
        separar_por_similaridade, LIMIAR_SIMILARIDADE
    )
    idx_com, idx_sem, _ = separar_por_similaridade(
        list(X_train_cv), list(X_test), limiar=LIMIAR_SIMILARIDADE
    )
    logger.info(f"Teste: {len(idx_com)} frases com análogo próximo, {len(idx_sem)} sem")

    test_results = {}

    for name, pipeline in BASELINES.items():
        logger.info(f"\n--- Teste Final: {name} ---")

        from sklearn.base import clone
        model = clone(pipeline)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_tst)
        y_pred_dist = model.predict(X_dist)
        
        f1 = f1_score(y_tst, y_pred, average='macro')
        acc = accuracy_score(y_tst, y_pred)
        prec = precision_score(y_tst, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_tst, y_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_tst, y_pred).tolist()
        report = classification_report(y_tst, y_pred, target_names=CLASS_NAMES)

        test_results[name] = {
            'f1': f1,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            # Por classe, em formato numérico: o relatório em texto acima é
            # legível, mas não serve para gerar as figuras do artigo.
            'f1_per_class': f1_score(y_tst, y_pred, average=None, zero_division=0).tolist(),
            'precision_per_class': precision_score(y_tst, y_pred, average=None, zero_division=0).tolist(),
            'recall_per_class': recall_score(y_tst, y_pred, average=None, zero_division=0).tolist(),
            'confusion_matrix': cm,
            'classification_report': report,
            'por_sentenca_distinta': {
                'n_textos_distintos': len(y_dist),
                'f1': f1_score(y_dist, y_pred_dist, average='macro'),
                'accuracy': accuracy_score(y_dist, y_pred_dist),
            },
            'por_similaridade': {
                'limiar': LIMIAR_SIMILARIDADE,
                'com_analogo': {
                    'n': len(idx_com),
                    'f1': f1_score(y_tst[idx_com], y_pred[idx_com], average='macro'),
                    'accuracy': accuracy_score(y_tst[idx_com], y_pred[idx_com]),
                } if idx_com else None,
                'sem_analogo': {
                    'n': len(idx_sem),
                    'f1': f1_score(y_tst[idx_sem], y_pred[idx_sem], average='macro'),
                    'accuracy': accuracy_score(y_tst[idx_sem], y_pred[idx_sem]),
                } if idx_sem else None,
            },
        }

        logger.info(f"  F1={f1:.4f} | Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f}")
        logger.info(f"  (textos distintos, n={len(y_dist)}): "
                    f"F1={test_results[name]['por_sentenca_distinta']['f1']:.4f}")
        sim = test_results[name]['por_similaridade']
        if sim['sem_analogo']:
            logger.info(f"  (sem análogo próximo, n={sim['sem_analogo']['n']}): "
                        f"F1={sim['sem_analogo']['f1']:.4f}")
        logger.info(f"\n{report}")
        logger.info(f"  Matriz de Confusão:\n{np.array(cm)}")
    
    return test_results


def load_bertimbau_reference():
    """
    Lê o resultado mais recente do grid search do BERTimbau, para que a
    comparação use o número realmente produzido pelo treino — e não uma
    constante copiada no código.

    Returns:
        Dict com as métricas de CV do BERTimbau, ou None se não houver
        resultado de grid search salvo.
    """
    gridsearch_dir = str(experimento.dir_resultados('toxicity_gridsearch'))
    if not os.path.isdir(gridsearch_dir):
        return None

    arquivos = [
        os.path.join(gridsearch_dir, f)
        for f in os.listdir(gridsearch_dir) if f.endswith('.json')
    ]
    if not arquivos:
        return None

    mais_recente = max(arquivos, key=os.path.getmtime)
    with open(mais_recente, 'r', encoding='utf-8') as f:
        data = json.load(f)

    best = data.get('best_config')
    if not best:
        return None

    # 'cv_f1_std' da busca em 3 estágios é o desvio ENTRE SEMENTES; o desvio
    # entre folds (o das baselines) é o 'std_f1' de cada semente. Os dois vão
    # separados para que nenhuma tabela misture os tipos de dispersão (M3a).
    por_semente = best.get('por_semente') or []
    dp_folds = [s['std_f1'] for s in por_semente if s.get('std_f1') is not None]
    return {
        'source_file': os.path.basename(mais_recente),
        'config': best.get('config'),
        'cv_f1_mean': best.get('mean_f1'),
        'cv_f1_std': best.get('std_f1'),
        'cv_f1_std_tipo': 'entre sementes' if por_semente else 'entre folds',
        'cv_f1_dp_entre_folds_media_sementes': (float(np.mean(dp_folds)) if dp_folds
                                                else best.get('std_f1')),
        'cv_accuracy_mean': best.get('mean_accuracy')
    }


def print_comparison_table(cv_results, test_results, bertimbau_ref=None):
    """Imprime tabela comparativa formatada."""

    # --- Tabela de Cross-Validation ---
    print("\n" + "=" * 80)
    print("RESULTADOS — CROSS-VALIDATION (5 FOLDS)")
    print("=" * 80)
    print(f"{'Modelo':<25} {'F1 (macro)':<18} {'Accuracy':<18} {'Precision':<18} {'Recall':<18}")
    print("-" * 80)
    
    # Ordena por F1 decrescente
    sorted_models = sorted(cv_results.items(), key=lambda x: x[1]['f1_mean'], reverse=True)
    
    for name, metrics in sorted_models:
        f1_str = f"{metrics['f1_mean']:.4f} ±{metrics['f1_std']:.4f}"
        acc_str = f"{metrics['accuracy_mean']:.4f} ±{metrics['accuracy_std']:.4f}"
        prec_str = f"{metrics['precision_mean']:.4f} ±{metrics['precision_std']:.4f}"
        rec_str = f"{metrics['recall_mean']:.4f} ±{metrics['recall_std']:.4f}"
        print(f"{name:<25} {f1_str:<18} {acc_str:<18} {prec_str:<18} {rec_str:<18}")
    
    # Adiciona BERTimbau para comparação
    print("-" * 80)
    if bertimbau_ref:
        ref_str = f"{bertimbau_ref['cv_f1_mean']:.4f} ±{bertimbau_ref['cv_f1_std']:.4f}"
        # A busca nem sempre registra a acurácia média de CV: quando não registra,
        # a coluna fica vazia em vez de derrubar a tabela inteira.
        acc = bertimbau_ref.get('cv_accuracy_mean')
        acc_str = f"{acc:.4f}" if acc is not None else "—"
        print(f"{'BERTimbau (ref.)':<25} {ref_str:<18} {acc_str:<18}")
    else:
        print(f"{'BERTimbau (ref.)':<25} (nenhum grid search salvo — rode train_toxicity_gridsearch)")
    print("=" * 80)

    # --- Tabela do Conjunto de Teste ---
    print("\n" + "=" * 80)
    print("RESULTADOS — CONJUNTO DE TESTE")
    print("=" * 80)
    print(f"{'Modelo':<25} {'F1 (macro)':<15} {'Accuracy':<15} {'Precision':<15} {'Recall':<15}")
    print("-" * 80)
    
    sorted_test = sorted(test_results.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    for name, metrics in sorted_test:
        print(f"{name:<25} {metrics['f1']:<15.4f} {metrics['accuracy']:<15.4f} {metrics['precision']:<15.4f} {metrics['recall']:<15.4f}")
    
    print("=" * 80)


def save_results(cv_results, test_results, split_sizes, bertimbau_ref=None):
    """Salva resultados em arquivo JSON."""
    output_dir = str(experimento.dir_resultados())
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"baselines_toxicity_{timestamp}.json")
    
    # Remove fold_details para a versão salva (muito verboso)
    cv_summary = {}
    for name, metrics in cv_results.items():
        cv_summary[name] = {k: v for k, v in metrics.items() if k != 'fold_details'}
    
    output = {
        'timestamp': timestamp,
        'methodology': {
            'features': 'TF-IDF (unigrams + bigrams, max_features=10000, sublinear_tf=True)',
            'cv_folds': K_FOLDS,
            'cv_scheme': ('folds gravados, agrupados por vídeo (data/particoes/particao_video.csv)'
                          if experimento.particao() == 'video' else
                          'StratifiedGroupKFold (grupo = texto normalizado)'),
            'experimento': experimento.descricao(),
            'random_state': RANDOM_STATE,
            **split_sizes
        },
        'cross_validation_results': cv_summary,
        'test_results': test_results,
        'bertimbau_reference': bertimbau_ref
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResultados salvos em: {output_file}")
    return output_file


def main():
    """Executa todas as baselines e gera relatório comparativo."""
    
    print("=" * 80)
    print("BASELINES CLÁSSICAS — DETECÇÃO DE TOXICIDADE")
    print("Comparação com BERTimbau fine-tuned")
    print("=" * 80)
    
    # 1. Carrega dados (mesma divisão do grid search)
    logger.info("Carregando dados...")
    (X_train_cv, y_train_cv, groups), (X_test, y_test) = get_data_for_cv_and_test(
        test_size=0.20, return_groups=True
    )

    logger.info(f"Treino (CV): {len(X_train_cv)} amostras")
    logger.info(f"Teste: {len(X_test)} amostras")

    split_sizes = {
        'corpus_total': len(X_train_cv) + len(X_test),
        'train_size': len(X_train_cv),
        'test_size': len(X_test)
    }

    # 2. Cross-validation (5 folds, mesmos grupos do grid search)
    logger.info("\n>>> Iniciando Cross-Validation...")
    cv_results = evaluate_with_cv(X_train_cv, y_train_cv, groups=groups)

    # 3. Avaliação no conjunto de teste
    logger.info("\n>>> Avaliando no conjunto de teste...")
    test_results = evaluate_on_test(X_train_cv, y_train_cv, X_test, y_test)

    # 4. Tabela comparativa
    bertimbau_ref = load_bertimbau_reference()
    print_comparison_table(cv_results, test_results, bertimbau_ref)

    # 5. Salva resultados
    output_file = save_results(cv_results, test_results, split_sizes, bertimbau_ref)

    print(f"\n[OK] Resultados salvos em: {output_file}")
    if bertimbau_ref:
        print(f"[OK] Referência BERTimbau ({bertimbau_ref['source_file']}): "
              f"F1(CV) = {bertimbau_ref['cv_f1_mean']:.4f} ±{bertimbau_ref['cv_f1_std']:.4f}")


if __name__ == "__main__":
    main()
