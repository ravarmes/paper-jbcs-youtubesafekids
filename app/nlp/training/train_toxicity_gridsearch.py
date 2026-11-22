"""
Script de treinamento com Grid Search para Detecção de Toxicidade.

Metodologia Científica:
- Grid Search com Successive Halving (2 estágios)
- Cross-Validation (K=5) para cada configuração
- Oversampling apenas no conjunto de treino de cada fold
- Métricas: F1 Macro, F1 Weighted, Precision, Recall
- Holdout de 20% para teste final (nunca usado no tuning)

Modos de execução:
- FAST: Grid Search reduzido (1 configuração) para validar pipeline
- FULL: Grid Search completo (72 configurações) - uso não recomendado
- OPTIMIZED: Grid Search com Successive Halving (recomendado para artigo)
  - Estágio 1: Todas configs × 2 folds × 3 épocas (triagem rápida)
  - Estágio 2: Top-N configs × 5 folds × épocas completas (avaliação final)

Opções de pré-processamento:
- --preprocess: Aplica limpeza de texto e lematização (conforme TCC)
- Sem flag: Usa texto original (apenas tokenização do BERT)
"""

import logging
import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product
from pathlib import Path
from sklearn.utils import resample
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    precision_recall_fscore_support
)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp.models.bertimbau_toxicity import BertimbauToxicity
from app.nlp.datasets.prepare_data_toxicity import get_data_for_cv_and_test
from app.nlp.utils.data_utils import get_kfold_split
from app.nlp.utils.preprocessing import preprocess_batch
from app.nlp.config import PATHS, get_task_config
from app.nlp import experimento

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURAÇÕES DO GRID SEARCH
# ============================================================================

# MODO FAST: Grid Search reduzido para validação rápida do pipeline
GRID_FAST = {
    'epochs': [3],
    'batch_size': [16],
    'learning_rate': [5e-5],
    'max_length': [128]
}

# MODO FULL: Grid Search ampliado original (NÃO RECOMENDADO - ~56 dias)
GRID_FULL = {
    'epochs': [3, 5, 8],
    'batch_size': [4, 8, 16],
    'learning_rate': [1e-5, 2e-5, 3e-5, 5e-5],
    'max_length': [128, 256]
}

# MODO OPTIMIZED: Grid cientificamente otimizado (base para Successive Halving)
# Justificativa: Devlin et al. (2019) recomendam batch_size ∈ {8, 16, 32},
# lr ∈ {2e-5, 3e-5, 5e-5}, epochs ∈ {2, 3, 4} para fine-tuning de BERT.
# max_length=128 é suficiente para comentários curtos do YouTube.
GRID_OPTIMIZED = {
    'epochs': [3, 5],
    'batch_size': [8, 16],
    'learning_rate': [2e-5, 3e-5, 5e-5],
    'max_length': [128]
}

# MODO SEARCH: espaço ampliado, percorrido por busca aleatória.
# Com 7 dimensões o produto cartesiano passa de 1.700 combinações; amostrar ao
# acaso cobre melhor cada dimensão isolada do que uma grade grosseira com o
# mesmo orçamento (Bergstra & Bengio, 2012). Além dos hiperparâmetros de
# otimização, entram na busca três decisões de estratégia que antes eram fixas:
#   - loss: 'focal' ou entropia cruzada simples
#   - oversample: replicar as classes minoritárias no treino, ou não
#   - llrd: decaimento da taxa de aprendizado por camada
ESPACO_BUSCA = {
    'epochs': [3, 4, 5, 8],
    'batch_size': [8, 16, 32],
    'learning_rate': [1e-5, 2e-5, 3e-5, 5e-5],
    'max_length': [128],
    'warmup_ratio': [0.06, 0.1],
    'weight_decay': [0.01, 0.1],
    'llrd': [1.0, 0.95, 0.9],
    'loss_type': ['focal', 'ce'],
    'label_smoothing': [0.0, 0.1],
    'oversample': [True, False],
}

N_AMOSTRAS_BUSCA = 32     # configurações sorteadas para a triagem

# Configurações para Successive Halving (modo optimized)
STAGE1_FOLDS = 2          # Folds reduzidos para triagem rápida
STAGE1_EPOCHS = 3         # Épocas fixas no estágio 1 (apenas para ranking)
TOP_N_CONFIGS = 6         # Quantas configs avançam para o estágio 2

# Successive Halving de três estágios (modo search)
SEARCH_STAGE1_FOLDS = 2
SEARCH_STAGE1_EPOCHS = 3
SEARCH_TOP_STAGE2 = 10    # avançam da triagem para a avaliação completa
SEARCH_TOP_STAGE3 = 3     # avançam para o teste de estabilidade entre sementes
SEARCH_SEEDS = [42, 7, 2024]

K_FOLDS = 5

# Fração do treino de cada fold reservada como dev interno para escolher a época
DEV_INTERNO = 0.12

# Mapeamento de parâmetros do Grid para argumentos do Trainer
GRID_PARAM_MAPPING = {
    'epochs': 'num_train_epochs',
    'batch_size': 'per_device_train_batch_size',
    'learning_rate': 'learning_rate',
    'max_length': 'max_length',
    'warmup_ratio': 'warmup_ratio',
    'weight_decay': 'weight_decay',
    'llrd': 'llrd',
    'loss_type': 'loss_type',
    'label_smoothing': 'label_smoothing',
}


def apply_oversampling(train_texts, train_labels):
    """
    Balanceia classes minoritárias via Random Oversampling.
    Aplicado APENAS no conjunto de treino para evitar data leakage.
    
    Classes: 0=Nenhuma, 1=Leve, 2=Severa
    """
    df = pd.DataFrame({'text': train_texts, 'label': train_labels})
    
    # Separa por classe
    df_class_0 = df[df['label'] == 0]  # Nenhuma
    df_class_1 = df[df['label'] == 1]  # Leve
    df_class_2 = df[df['label'] == 2]  # Severa
    
    # Define alvo como a classe majoritária
    max_count = max(len(df_class_0), len(df_class_1), len(df_class_2))
    
    # Upsample das classes minoritárias
    df_0_up = resample(df_class_0, replace=True, n_samples=max_count, random_state=42)
    df_1_up = resample(df_class_1, replace=True, n_samples=max_count, random_state=42)
    df_2_up = resample(df_class_2, replace=True, n_samples=max_count, random_state=42)
    
    df_balanced = pd.concat([df_0_up, df_1_up, df_2_up])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_balanced['text'].tolist(), df_balanced['label'].tolist()


def amostrar_configuracoes(espaco, n_amostras, random_state=42):
    """
    Sorteia `n_amostras` configurações distintas do espaço de busca.

    Busca aleatória em vez de grade: com muitas dimensões, sortear cobre melhor
    cada hiperparâmetro isolado do que uma grade grosseira de mesmo custo
    (Bergstra & Bengio, 2012).
    """
    rng = np.random.RandomState(random_state)
    nomes = list(espaco.keys())
    vistas, configs = set(), []

    total_possivel = 1
    for valores in espaco.values():
        total_possivel *= len(valores)
    alvo = min(n_amostras, total_possivel)

    while len(configs) < alvo:
        escolha = tuple(valores[rng.randint(len(valores))] for valores in espaco.values())
        if escolha in vistas:
            continue
        vistas.add(escolha)
        configs.append(dict(zip(nomes, escolha)))

    return configs


def separar_dev_interno(X, y, fracao=DEV_INTERNO, random_state=42, grupos=None):
    """
    Separa um dev interno do treino do fold.

    A escolha da melhor época precisa de um conjunto de validação; usar o
    próprio conjunto de validação do fold para isso tornaria a métrica
    reportada otimista, porque o mesmo dado escolheria e avaliaria o modelo.

    Na partição por vídeo (`grupos` = video_id), o dev é formado por vídeos
    inteiros (GroupShuffleSplit, `fracao` dos vídeos): o early stopping não vê
    frases de vídeos que estão no treino. A separação não usa os rótulos, de modo
    que é a mesma nas duas rotulagens. Sem grupos, mantém a divisão estratificada
    por frase da versão submetida.
    """
    from app.nlp import experimento
    if grupos is not None and experimento.particao() == 'video':
        from sklearn.model_selection import GroupShuffleSplit
        divisor = GroupShuffleSplit(n_splits=1, test_size=fracao, random_state=random_state)
        tr, dev = next(divisor.split(X, y, groups=grupos))
        X, y = list(X), list(y)
        return ([X[i] for i in tr], [y[i] for i in tr],
                [X[i] for i in dev], [y[i] for i in dev])

    from sklearn.model_selection import train_test_split
    X_tr, X_dev, y_tr, y_dev = train_test_split(
        X, y, test_size=fracao, random_state=random_state, stratify=y
    )
    return X_tr, y_tr, X_dev, y_dev


def _train_single_fold(config, X_train_cv, y_train_cv, train_idx, val_idx,
                       config_idx, fold, total_folds, override_epochs=None,
                       groups=None):
    """
    Treina um único fold de uma configuração.
    
    Args:
        config: Dict com os hiperparâmetros
        X_train_cv: Array com todos os textos de CV
        y_train_cv: Array com todos os labels de CV
        train_idx: Índices do treino neste fold
        val_idx: Índices da validação neste fold
        config_idx: Índice da configuração (para logging)
        fold: Número do fold atual (0-indexed)
        total_folds: Total de folds
        override_epochs: Se fornecido, sobrescreve o número de épocas da config
        
    Returns:
        Dict com métricas do fold ou None em caso de erro
    """
    curr_fold = fold + 1
    logger.info(f"\n>>> [Fold {curr_fold}/{total_folds}]")

    # Separação do fold
    X_fold_train = X_train_cv[train_idx]
    X_fold_val = X_train_cv[val_idx]
    y_fold_train = y_train_cv[train_idx]
    y_fold_val = y_train_cv[val_idx]

    effective_config = config.copy()
    if override_epochs is not None:
        effective_config['epochs'] = override_epochs

    usar_melhor_epoca = bool(effective_config.get('usar_melhor_epoca', True))
    seed = int(effective_config.get('seed', 42))

    # Dev interno para escolher a época; o conjunto de validação do fold fica
    # intocado e é usado só para reportar a métrica.
    if usar_melhor_epoca:
        X_tr, y_tr, X_dev, y_dev = separar_dev_interno(
            X_fold_train.tolist(), y_fold_train.tolist(), random_state=seed,
            grupos=None if groups is None else np.asarray(groups)[train_idx]
        )
    else:
        X_tr, y_tr = X_fold_train.tolist(), y_fold_train.tolist()
        X_dev, y_dev = X_fold_val.tolist(), y_fold_val.tolist()

    # Oversampling apenas no treino, e só quando a configuração pedir
    if effective_config.get('oversample', True):
        X_tr, y_tr = apply_oversampling(X_tr, y_tr)
        rotulo_treino = 'balanceado'
    else:
        rotulo_treino = 'sem balanceamento'

    logger.info(f"  Treino ({rotulo_treino}): {len(X_tr)} | dev interno: {len(X_dev)} "
                f"| validação do fold: {len(X_fold_val)}")

    # Instancia modelo limpo (o checkpoint base pode variar na busca)
    base_model = effective_config.get('base_model')
    model = BertimbauToxicity(config={'base_model': base_model} if base_model else None)

    # Prepara kwargs com hiperparâmetros
    train_kwargs = {}
    for p_name, p_val in effective_config.items():
        if p_name in GRID_PARAM_MAPPING:
            target_arg = GRID_PARAM_MAPPING[p_name]
            train_kwargs[target_arg] = p_val
            # Ajusta batch size de validação também
            if p_name == 'batch_size':
                train_kwargs['per_device_eval_batch_size'] = p_val

    train_kwargs.setdefault('loss_type', 'focal')
    train_kwargs.setdefault('label_smoothing', 0.1)
    train_kwargs['focal_gamma'] = float(effective_config.get('focal_gamma', 2.0))
    train_kwargs['usar_melhor_epoca'] = usar_melhor_epoca
    train_kwargs['seed'] = seed

    try:
        epoch_label = f"ep={effective_config['epochs']}"
        results = model.train_model(
            train_texts=X_tr,
            train_labels=y_tr,
            val_texts=X_dev,
            val_labels=y_dev,
            config_name='default',
            experiment_name=f'TOX_grid_{config_idx}_fold_{curr_fold}_{epoch_label}',
            save_artifacts=False,
            **train_kwargs
        )

        # Métrica do fold: avaliada no conjunto de validação, que não participou
        # nem do treino nem da escolha da época.
        metrics = avaliar_no_fold(model, X_fold_val.tolist(), y_fold_val.tolist())
        metrics['fold'] = curr_fold
        metrics['dev_f1'] = results.get('final_metrics', {}).get('eval_f1')

        logger.info(f"  [Fold {curr_fold}] F1 (validação do fold): {metrics['eval_f1']:.4f}")

        return metrics

    except Exception as e:
        logger.error(f"  Erro no fold {curr_fold}: {e}")
        return {'fold': curr_fold, 'error': str(e)}
    finally:
        # Libera a VRAM antes do próximo fold
        try:
            import torch
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass


def _treinar_fold_ou_abortar(*args, **kwargs):
    """
    Treina um fold; se falhar, tenta mais uma vez e, falhando de novo, aborta.

    Antes, um fold com erro (ex.: falta de memória na GPU) era descartado e a
    configuração recebia a média dos folds restantes -- uma nota calculada sobre
    outra amostra, comparada com as das demais configurações como se fosse a
    mesma coisa. Uma busca assim escolhe errado em silêncio. Agora o erro
    persistente interrompe a rodada, que deve ser refeita.
    """
    import time
    metrics = _train_single_fold(*args, **kwargs)
    if 'error' not in metrics:
        return metrics
    logger.warning(f"  Fold {metrics.get('fold')} falhou ({metrics['error']}); "
                   "nova tentativa em 30 s")
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    time.sleep(30)
    metrics = _train_single_fold(*args, **kwargs)
    if 'error' in metrics:
        raise RuntimeError(f"fold {metrics.get('fold')} falhou duas vezes: "
                           f"{metrics['error']} -- busca interrompida para não "
                           "comparar configurações avaliadas em folds diferentes")
    return metrics


def avaliar_no_fold(model, textos, rotulos, batch_size=64):
    """
    Avalia o modelo treinado no conjunto de validação do fold.

    Returns:
        Dict com as métricas no formato usado pelo restante do script
        (`eval_f1` = F1 macro).
    """
    import torch

    model.model.eval()
    device = model.device
    preds = []

    with torch.no_grad():
        for i in range(0, len(textos), batch_size):
            lote = textos[i:i + batch_size]
            entradas = model.tokenizer(
                lote, padding='max_length', truncation=True,
                max_length=model.max_length, return_tensors='pt'
            )
            entradas = {k: v.to(device) for k, v in entradas.items()}
            logits = model.model(**entradas).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())

    precisao, recall, f1, _ = precision_recall_fscore_support(
        rotulos, preds, average='macro', zero_division=0
    )
    return {
        'eval_f1': float(f1),
        'eval_accuracy': float(accuracy_score(rotulos, preds)),
        'eval_precision': float(precisao),
        'eval_recall': float(recall),
        'eval_loss': 0.0,
    }


def _evaluate_configs(combinations, param_names, X_train_cv, y_train_cv,
                      n_folds, override_epochs=None, stage_label="", groups=None):
    """
    Avalia uma lista de configurações com cross-validation.

    Args:
        combinations: Lista de tuplas com valores dos hiperparâmetros
        param_names: Nomes dos hiperparâmetros
        X_train_cv: Array de textos
        y_train_cv: Array de labels
        n_folds: Número de folds para cross-validation
        override_epochs: Se fornecido, sobrescreve épocas em todas as configs
        stage_label: Label para logging (ex: "ESTÁGIO 1", "ESTÁGIO 2")
        groups: Grupos (texto normalizado) para folds sem vazamento de frases

    Returns:
        Lista de dicts com resultados por configuração
    """
    all_results = []
    
    for config_idx, params in enumerate(combinations, 1):
        config = dict(zip(param_names, params))
        logger.info(f"\n{'='*70}")
        if stage_label:
            logger.info(f"{stage_label} — CONFIGURAÇÃO {config_idx}/{len(combinations)}")
        else:
            logger.info(f"CONFIGURAÇÃO {config_idx}/{len(combinations)}")
        
        epoch_display = override_epochs if override_epochs else config.get('epochs', '?')
        logger.info(f"Parâmetros: {config} (épocas efetivas: {epoch_display})")
        logger.info(f"{'='*70}")
        
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(
            get_kfold_split(X_train_cv, y_train_cv, n_splits=n_folds, groups=groups)
        ):
            metrics = _treinar_fold_ou_abortar(
                config, X_train_cv, y_train_cv,
                train_idx, val_idx,
                config_idx, fold, n_folds,
                override_epochs=override_epochs, groups=groups
            )
            fold_metrics.append(metrics)
        
        # Calcula métricas médias da configuração
        valid_metrics = [m for m in fold_metrics if 'error' not in m]
        if valid_metrics:
            df_metrics = pd.DataFrame(valid_metrics)
            
            config_result = {
                'config': config,
                'config_idx': config_idx,
                'n_folds_ok': len(valid_metrics),
                'mean_f1': df_metrics['eval_f1'].mean() if 'eval_f1' in df_metrics else 0,
                'std_f1': df_metrics['eval_f1'].std() if 'eval_f1' in df_metrics else 0,
                'mean_accuracy': df_metrics['eval_accuracy'].mean() if 'eval_accuracy' in df_metrics else 0,
                'mean_loss': df_metrics['eval_loss'].mean() if 'eval_loss' in df_metrics else 0,
                'fold_metrics': fold_metrics
            }
            
            all_results.append(config_result)
            
            logger.info(f"\nMédia Config {config_idx}: "
                        f"F1={config_result['mean_f1']:.4f} "
                        f"(+/- {config_result['std_f1']:.4f})")
        else:
            logger.warning(f"Configuração {config_idx} falhou em todos os folds!")
    
    return all_results


def _avaliar_lista_de_configs(configs, X_train_cv, y_train_cv, n_folds,
                              override_epochs=None, stage_label="", groups=None):
    """
    Versão de `_evaluate_configs` que recebe dicionários de configuração já
    montados (a busca aleatória não trabalha com produto cartesiano).
    """
    resultados = []

    for idx, config in enumerate(configs, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"{stage_label} — CONFIGURAÇÃO {idx}/{len(configs)}")
        epocas = override_epochs if override_epochs else config.get('epochs', '?')
        logger.info(f"Parâmetros: {config} (épocas efetivas: {epocas})")
        logger.info(f"{'='*70}")

        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(
            get_kfold_split(X_train_cv, y_train_cv, n_splits=n_folds, groups=groups)
        ):
            fold_metrics.append(_treinar_fold_ou_abortar(
                config, X_train_cv, y_train_cv, train_idx, val_idx,
                idx, fold, n_folds, override_epochs=override_epochs, groups=groups
            ))

        validos = [m for m in fold_metrics if 'error' not in m]
        if not validos:
            logger.warning(f"Configuração {idx} falhou em todos os folds!")
            continue

        df = pd.DataFrame(validos)
        resultado = {
            'config': config,
            'config_idx': idx,
            'n_folds_ok': len(validos),
            'mean_f1': float(df['eval_f1'].mean()),
            'std_f1': float(df['eval_f1'].std()) if len(df) > 1 else 0.0,
            'mean_accuracy': float(df['eval_accuracy'].mean()),
            'fold_metrics': fold_metrics,
        }
        resultados.append(resultado)
        logger.info(f"\nMédia Config {idx}: F1={resultado['mean_f1']:.4f} "
                    f"(+/- {resultado['std_f1']:.4f})")

    return resultados


def run_random_search(preprocess=False, n_amostras=N_AMOSTRAS_BUSCA):
    """
    Busca aleatória sobre o espaço ampliado, com Successive Halving em três
    estágios.

    Estágio 1 — triagem: `n_amostras` configurações × 2 folds × 3 épocas.
    Estágio 2 — avaliação completa: Top-12 × 5 folds, épocas próprias.
    Estágio 3 — estabilidade: Top-3 × 5 folds × 3 sementes; vence a de maior
                F1 médio entre sementes, critério que penaliza configurações
                boas por acaso em uma inicialização específica.

    Toda a seleção usa exclusivamente os folds de validação cruzada; o conjunto
    de teste retido não é consultado em nenhum momento.
    """
    logger.info("=" * 70)
    logger.info("BUSCA ALEATÓRIA + SUCCESSIVE HALVING (3 ESTÁGIOS) — TOXICIDADE")
    logger.info("=" * 70)

    configs = amostrar_configuracoes(ESPACO_BUSCA, n_amostras)
    logger.info(f"Espaço de busca: {ESPACO_BUSCA}")
    logger.info(f"Configurações sorteadas: {len(configs)}")

    output_dir = experimento.dir_resultados('toxicity_gridsearch')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parcial = output_dir / f'busca_parcial_{timestamp}.json'

    def salvar_parcial(**partes):
        """Grava o que já foi apurado, para a busca não se perder se algo falhar."""
        with open(parcial, 'w', encoding='utf-8') as f:
            json.dump(partes, f, indent=2, default=str, ensure_ascii=False)

    (X_list, y_list, groups_list), (X_test_final, y_test_final) = \
        get_data_for_cv_and_test(test_size=0.20, return_groups=True)

    if preprocess:
        logger.info("Aplicando pré-processamento (limpeza + lematização)...")
        X_list = preprocess_batch(X_list, apply_cleaning=True, apply_lemmatization=True)
        X_test_final = preprocess_batch(X_test_final, apply_cleaning=True,
                                        apply_lemmatization=True, show_progress=False)

    X_train_cv = np.array(X_list)
    y_train_cv = np.array(y_list)
    groups = np.array(groups_list)

    # --- Estágio 1: triagem -------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info(f"ESTÁGIO 1: TRIAGEM ({len(configs)} configs × {SEARCH_STAGE1_FOLDS} folds "
                f"× {SEARCH_STAGE1_EPOCHS} épocas)")
    logger.info("=" * 70)

    stage1 = _avaliar_lista_de_configs(
        configs, X_train_cv, y_train_cv, n_folds=SEARCH_STAGE1_FOLDS,
        override_epochs=SEARCH_STAGE1_EPOCHS, stage_label="ESTÁGIO 1", groups=groups
    )
    stage1.sort(key=lambda r: r['mean_f1'], reverse=True)
    salvar_parcial(stage1=stage1)

    logger.info("\n" + "=" * 70)
    logger.info("RANKING DO ESTÁGIO 1 (TRIAGEM)")
    logger.info("=" * 70)
    for pos, r in enumerate(stage1, 1):
        marca = " ✓ AVANÇA" if pos <= SEARCH_TOP_STAGE2 else ""
        logger.info(f"  #{pos}: F1={r['mean_f1']:.4f} (+/- {r['std_f1']:.4f}) | "
                    f"{r['config']}{marca}")

    # --- Estágio 2: avaliação completa --------------------------------------
    top2 = [r['config'] for r in stage1[:SEARCH_TOP_STAGE2]]
    logger.info("\n" + "=" * 70)
    logger.info(f"ESTÁGIO 2: AVALIAÇÃO COMPLETA ({len(top2)} configs × {K_FOLDS} folds)")
    logger.info("=" * 70)

    stage2 = _avaliar_lista_de_configs(
        top2, X_train_cv, y_train_cv, n_folds=K_FOLDS,
        override_epochs=None, stage_label="ESTÁGIO 2", groups=groups
    )
    stage2.sort(key=lambda r: r['mean_f1'], reverse=True)
    salvar_parcial(stage1=stage1, stage2=stage2)

    # --- Estágio 3: estabilidade entre sementes -----------------------------
    top3 = [r['config'] for r in stage2[:SEARCH_TOP_STAGE3]]
    logger.info("\n" + "=" * 70)
    logger.info(f"ESTÁGIO 3: ESTABILIDADE ({len(top3)} configs × {len(SEARCH_SEEDS)} sementes)")
    logger.info("=" * 70)

    stage3 = []
    for config in top3:
        por_semente = []
        for semente in SEARCH_SEEDS:
            variante = dict(config, seed=semente)
            resultado = _avaliar_lista_de_configs(
                [variante], X_train_cv, y_train_cv, n_folds=K_FOLDS,
                override_epochs=None, stage_label=f"ESTÁGIO 3 (semente {semente})",
                groups=groups
            )
            if resultado:
                por_semente.append({'seed': semente, 'mean_f1': resultado[0]['mean_f1'],
                                    'std_f1': resultado[0]['std_f1']})

        if not por_semente:
            continue

        f1s = [s['mean_f1'] for s in por_semente]
        stage3.append({
            'config': config,
            'por_semente': por_semente,
            'mean_f1': float(np.mean(f1s)),
            'std_f1': float(np.std(f1s)),
            'pior_semente': float(np.min(f1s)),
        })
        logger.info(f"\n>>> Config {config}: F1 médio entre sementes = "
                    f"{stage3[-1]['mean_f1']:.4f} (±{stage3[-1]['std_f1']:.4f}), "
                    f"pior semente = {stage3[-1]['pior_semente']:.4f}")

    stage3.sort(key=lambda r: r['mean_f1'], reverse=True)
    melhor = stage3[0] if stage3 else (stage2[0] if stage2 else None)

    # --- Relatório e persistência ------------------------------------------
    print("\n" + "=" * 70)
    print("RESUMO — BUSCA ALEATÓRIA + SUCCESSIVE HALVING")
    print("=" * 70)
    print(f"\n--- Estágio 2 (Top-{SEARCH_TOP_STAGE2}, {K_FOLDS} folds) ---")
    print(pd.DataFrame([
        {**{k: v for k, v in r['config'].items() if k != 'max_length'},
         'F1_mean': round(r['mean_f1'], 4), 'F1_std': round(r['std_f1'], 4)}
        for r in stage2
    ]).to_string(index=False))

    if stage3:
        print(f"\n--- Estágio 3 (estabilidade entre {len(SEARCH_SEEDS)} sementes) ---")
        print(pd.DataFrame([
            {**{k: v for k, v in r['config'].items() if k != 'max_length'},
             'F1_medio': round(r['mean_f1'], 4),
             'F1_std_sementes': round(r['std_f1'], 4),
             'pior_semente': round(r['pior_semente'], 4)}
            for r in stage3
        ]).to_string(index=False))

    if melhor:
        print(f"\n{'='*70}\nCONFIGURAÇÃO VENCEDORA\n{'='*70}")
        print(f"Parâmetros: {melhor['config']}")
        print(f"F1 médio (validação cruzada): {melhor['mean_f1']:.4f}")

    results_file = output_dir / f'gridsearch_optimized_search_{timestamp}.json'

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'mode': 'search',
            'method': 'random_search + successive_halving (3 estágios)',
            'espaco_busca': ESPACO_BUSCA,
            'n_amostras': len(configs),
            'preprocessing': preprocess,
            'stage1': {'folds': SEARCH_STAGE1_FOLDS, 'epochs_override': SEARCH_STAGE1_EPOCHS,
                       'results': stage1},
            'stage2': {'folds': K_FOLDS, 'top_n': SEARCH_TOP_STAGE2, 'results': stage2},
            'stage3': {'folds': K_FOLDS, 'seeds': SEARCH_SEEDS, 'results': stage3},
            'best_config': melhor,
            'test_set_size': len(X_test_final),
            'experimento': experimento.descricao(),
        }, f, indent=2, default=str, ensure_ascii=False)

    logger.info(f"\nResultados salvos em: {results_file}")

    best_model_path = None
    if melhor:
        try:
            best_model_path = train_final_best_model(
                melhor, X_train_cv, y_train_cv, X_test_final, y_test_final, groups=groups
            )
        except Exception as e:
            logger.error(f"Erro ao treinar e localizar modelo final: {e}")

    return stage2, melhor, best_model_path


def run_grid_search_optimized(preprocess=False):
    """
    Executa Grid Search Otimizado com Successive Halving (2 estágios).
    
    Estágio 1: Triagem rápida (2 folds, 3 épocas) de todas as configurações
    Estágio 2: Avaliação completa (5 folds, épocas originais) das Top-N configs
    
    Referência metodológica:
    - Jamieson & Talwalkar (2016). "Non-stochastic Best Arm Identification 
      and Hyperparameter Optimization" (Successive Halving)
    
    Args:
        preprocess: Se True, aplica limpeza + lematização nos textos
    """
    grid = GRID_OPTIMIZED
    preprocess_label = "COM" if preprocess else "SEM"
    
    logger.info("=" * 70)
    logger.info("GRID SEARCH OTIMIZADO (SUCCESSIVE HALVING) — TOXICIDADE")
    logger.info(f"PRÉ-PROCESSAMENTO: {preprocess_label} (limpeza + lematização)")
    logger.info("=" * 70)
    logger.info(f"Grid base: {grid}")
    
    # Gera todas as combinações
    param_names = list(grid.keys())
    param_values = list(grid.values())
    combinations = list(product(*param_values))
    
    logger.info(f"Total de configurações: {len(combinations)}")
    logger.info(f"Estágio 1: {len(combinations)} configs × {STAGE1_FOLDS} folds × {STAGE1_EPOCHS} épocas")
    logger.info(f"Estágio 2: Top-{TOP_N_CONFIGS} configs × {K_FOLDS} folds × épocas originais")
    stage1_treinos = len(combinations) * STAGE1_FOLDS
    stage2_treinos = TOP_N_CONFIGS * K_FOLDS
    logger.info(f"Total estimado de treinos: {stage1_treinos} + {stage2_treinos} = {stage1_treinos + stage2_treinos}")
    
    # Carrega dados (80% CV / 20% Teste), com grupos para folds sem vazamento
    (X_train_cv_list, y_train_cv_list, groups_list), (X_test_final, y_test_final) = \
        get_data_for_cv_and_test(test_size=0.20, return_groups=True)

    # Aplica pré-processamento se solicitado
    if preprocess:
        logger.info("Aplicando pré-processamento (limpeza + lematização)...")
        X_train_cv_list = preprocess_batch(X_train_cv_list, apply_cleaning=True, apply_lemmatization=True)
        X_test_final = preprocess_batch(X_test_final, apply_cleaning=True, apply_lemmatization=True, show_progress=False)

    X_train_cv = np.array(X_train_cv_list)
    y_train_cv = np.array(y_train_cv_list)
    groups = np.array(groups_list)

    # ===========================================================================
    # ESTÁGIO 1: Triagem Rápida
    # ===========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ESTÁGIO 1: TRIAGEM RÁPIDA")
    logger.info(f"Configurações: {len(combinations)} | Folds: {STAGE1_FOLDS} | Épocas: {STAGE1_EPOCHS}")
    logger.info("=" * 70)
    
    stage1_results = _evaluate_configs(
        combinations, param_names, X_train_cv, y_train_cv,
        n_folds=STAGE1_FOLDS,
        override_epochs=STAGE1_EPOCHS,
        stage_label="ESTÁGIO 1",
        groups=groups
    )
    
    # Ranking por F1 médio
    stage1_results.sort(key=lambda x: x['mean_f1'], reverse=True)
    
    logger.info("\n" + "=" * 70)
    logger.info("RANKING DO ESTÁGIO 1 (TRIAGEM)")
    logger.info("=" * 70)
    for rank, result in enumerate(stage1_results, 1):
        marker = " ✓ AVANÇA" if rank <= TOP_N_CONFIGS else ""
        logger.info(
            f"  #{rank}: F1={result['mean_f1']:.4f} "
            f"(+/- {result['std_f1']:.4f}) | "
            f"{result['config']}{marker}"
        )
    
    # Seleciona Top-N
    top_configs = stage1_results[:TOP_N_CONFIGS]
    top_combinations = [tuple(c['config'][p] for p in param_names) for c in top_configs]
    
    logger.info(f"\n{TOP_N_CONFIGS} configurações selecionadas para o Estágio 2.")
    
    # ===========================================================================
    # ESTÁGIO 2: Avaliação Completa
    # ===========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ESTÁGIO 2: AVALIAÇÃO COMPLETA")
    logger.info(f"Configurações: {len(top_combinations)} | Folds: {K_FOLDS} | Épocas: originais da config")
    logger.info("=" * 70)
    
    stage2_results = _evaluate_configs(
        top_combinations, param_names, X_train_cv, y_train_cv,
        n_folds=K_FOLDS,
        override_epochs=None,  # Usa as épocas originais da config
        stage_label="ESTÁGIO 2",
        groups=groups
    )
    
    # Ranking final
    stage2_results.sort(key=lambda x: x['mean_f1'], reverse=True)
    
    # Identifica melhor configuração
    best_config = stage2_results[0] if stage2_results else None
    
    # Relatório Final
    _print_final_report(stage1_results, stage2_results, best_config)
    
    # Salva resultados
    output_dir = experimento.dir_resultados('toxicity_gridsearch')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    preprocess_suffix = "_preprocess" if preprocess else "_raw"
    results_file = output_dir / f'gridsearch_optimized{preprocess_suffix}_{timestamp}.json'
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'mode': 'optimized',
            'method': 'successive_halving',
            'preprocessing': preprocess,
            'preprocessing_description': 'limpeza + lematização' if preprocess else 'texto original',
            'grid': grid,
            'stage1': {
                'folds': STAGE1_FOLDS,
                'epochs_override': STAGE1_EPOCHS,
                'total_configs': len(combinations),
                'results': stage1_results
            },
            'stage2': {
                'folds': K_FOLDS,
                'top_n': TOP_N_CONFIGS,
                'results': stage2_results
            },
            'best_config': best_config,
            'test_set_size': len(X_test_final)
        }, f, indent=2, default=str)
    
    logger.info(f"\nResultados salvos em: {results_file}")
    
    # Treina modelo final com a melhor config
    best_model_path = None
    if best_config:
        try:
            train_final_best_model(best_config, X_train_cv, y_train_cv, X_test_final, y_test_final)
            models_dir = experimento.dir_modelos()
            # Ignora as pastas de fold, que contêm apenas logs
            candidates = [p for p in models_dir.glob('TOX_*') if (p / 'config.json').exists()]
            if candidates:
                best_model_path = str(max(candidates, key=lambda p: p.stat().st_mtime))
        except Exception as e:
            logger.error(f"Erro ao treinar e localizar modelo final: {e}")

    return stage2_results, best_config, best_model_path


def _print_final_report(stage1_results, stage2_results, best_config):
    """Imprime relatório final consolidado."""
    print("\n" + "=" * 70)
    print("RESUMO FINAL DO GRID SEARCH OTIMIZADO (SUCCESSIVE HALVING)")
    print("=" * 70)
    
    print(f"\n--- Estágio 1 (Triagem): {len(stage1_results)} configs avaliadas ---")
    stage1_df = pd.DataFrame([
        {
            'config_idx': r['config_idx'],
            'epochs': r['config']['epochs'],
            'batch': r['config']['batch_size'],
            'lr': r['config']['learning_rate'],
            'max_len': r['config']['max_length'],
            'F1_mean': r['mean_f1'],
            'F1_std': r['std_f1'],
        }
        for r in stage1_results
    ])
    print(stage1_df.to_string(index=False))
    
    print(f"\n--- Estágio 2 (Avaliação Completa): Top-{TOP_N_CONFIGS} configs ---")
    stage2_df = pd.DataFrame([
        {
            'config_idx': r['config_idx'],
            'epochs': r['config']['epochs'],
            'batch': r['config']['batch_size'],
            'lr': r['config']['learning_rate'],
            'max_len': r['config']['max_length'],
            'F1_mean': r['mean_f1'],
            'F1_std': r['std_f1'],
            'Acc_mean': r['mean_accuracy']
        }
        for r in stage2_results
    ])
    print(stage2_df.to_string(index=False))
    
    if best_config:
        print(f"\n{'='*70}")
        print("MELHOR CONFIGURAÇÃO (VENCEDORA)")
        print(f"{'='*70}")
        print(f"Parâmetros: {best_config['config']}")
        print(f"F1-Score Médio: {best_config['mean_f1']:.4f} (+/- {best_config['std_f1']:.4f})")
        print(f"Accuracy Média: {best_config['mean_accuracy']:.4f}")
    
    print(f"\n{'='*70}")
    print(f"OBSERVAÇÃO IMPORTANTE:")
    print(f"Use evaluate_toxicity.py para avaliar o modelo final no conjunto de teste.")
    print(f"{'='*70}")


def run_grid_search(mode='fast', preprocess=False):
    """
    Executa Grid Search com Cross-Validation.
    
    Args:
        mode: 'fast' para validação rápida, 'full' para grid completo,
              'optimized' para successive halving (recomendado)
        preprocess: Se True, aplica limpeza + lematização nos textos
    """
    # Redireciona para os modos com Successive Halving
    if mode == 'optimized':
        return run_grid_search_optimized(preprocess=preprocess)
    if mode == 'search':
        return run_random_search(preprocess=preprocess)
    
    grid = GRID_FAST if mode == 'fast' else GRID_FULL
    preprocess_label = "COM" if preprocess else "SEM"
    
    logger.info("=" * 70)
    logger.info(f"INICIANDO GRID SEARCH ({mode.upper()}) - TOXICIDADE")
    logger.info(f"PRÉ-PROCESSAMENTO: {preprocess_label} (limpeza + lematização)")
    logger.info("=" * 70)
    logger.info(f"Hiperparâmetros a testar: {grid}")
    
    # Gera todas as combinações
    param_names = list(grid.keys())
    param_values = list(grid.values())
    combinations = list(product(*param_values))
    
    logger.info(f"Total de configurações: {len(combinations)}")
    logger.info(f"Folds por configuração: {K_FOLDS}")
    logger.info(f"Total de treinos: {len(combinations) * K_FOLDS}")
    
    # Carrega dados (80% CV / 20% Teste), com grupos para folds sem vazamento
    (X_train_cv_list, y_train_cv_list, groups_list), (X_test_final, y_test_final) = \
        get_data_for_cv_and_test(test_size=0.20, return_groups=True)

    # Aplica pré-processamento se solicitado
    if preprocess:
        logger.info("Aplicando pré-processamento (limpeza + lematização)...")
        X_train_cv_list = preprocess_batch(X_train_cv_list, apply_cleaning=True, apply_lemmatization=True)
        X_test_final = preprocess_batch(X_test_final, apply_cleaning=True, apply_lemmatization=True, show_progress=False)

    X_train_cv = np.array(X_train_cv_list)
    y_train_cv = np.array(y_train_cv_list)
    groups = np.array(groups_list)

    all_results = _evaluate_configs(
        combinations, param_names, X_train_cv, y_train_cv,
        n_folds=K_FOLDS, groups=groups
    )
    
    # Identifica melhor configuração
    best_config = None
    best_f1_macro = 0.0
    for result in all_results:
        if result['mean_f1'] > best_f1_macro:
            best_f1_macro = result['mean_f1']
            best_config = result
    
    # Relatório Final
    logger.info("\n" + "=" * 70)
    logger.info("RESUMO FINAL DO GRID SEARCH")
    logger.info("=" * 70)
    
    # Tabela de resultados
    results_df = pd.DataFrame([
        {
            'config_idx': r['config_idx'],
            'epochs': r['config']['epochs'],
            'batch': r['config']['batch_size'],
            'lr': r['config']['learning_rate'],
            'max_len': r['config']['max_length'],
            'F1_mean': r['mean_f1'],
            'F1_std': r['std_f1'],
            'Acc_mean': r['mean_accuracy']
        }
        for r in all_results
    ])
    
    print("\n--- Resultados por Configuração ---")
    print(results_df.to_string(index=False))
    
    if best_config:
        print(f"\n{'='*70}")
        print("MELHOR CONFIGURAÇÃO")
        print(f"{'='*70}")
        print(f"Parâmetros: {best_config['config']}")
        print(f"F1-Score Médio: {best_config['mean_f1']:.4f} (+/- {best_config['std_f1']:.4f})")
        print(f"Accuracy Média: {best_config['mean_accuracy']:.4f}")
    
    # Salva resultados
    output_dir = experimento.dir_resultados('toxicity_gridsearch')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    preprocess_suffix = "_preprocess" if preprocess else "_raw"
    results_file = output_dir / f'gridsearch_{mode}{preprocess_suffix}_{timestamp}.json'
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'mode': mode,
            'preprocessing': preprocess,
            'preprocessing_description': 'limpeza + lematização' if preprocess else 'texto original',
            'grid': grid,
            'results': all_results,
            'best_config': best_config,
            'test_set_size': len(X_test_final)
        }, f, indent=2, default=str)
    
    logger.info(f"\nResultados salvos em: {results_file}")
    
    print(f"\n{'='*70}")
    print(f"OBSERVAÇÃO IMPORTANTE:")
    print(f"O conjunto de TESTE FINAL ({len(X_test_final)} amostras, 20%) está reservado.")
    print(f"Use evaluate_toxicity.py para avaliar o modelo final neste conjunto.")
    print(f"{'='*70}")
    
    best_model_path = None
    if best_config:
        try:
            train_final_best_model(best_config, X_train_cv, y_train_cv, X_test_final, y_test_final)
            models_dir = experimento.dir_modelos()
            # Ignora as pastas de fold, que contêm apenas logs
            candidates = [p for p in models_dir.glob('TOX_*') if (p / 'config.json').exists()]
            if candidates:
                best_model_path = str(max(candidates, key=lambda p: p.stat().st_mtime))
        except Exception as e:
            logger.error(f"Erro ao treinar e localizar modelo final: {e}")
    return all_results, best_config, best_model_path


def train_final_best_model(best_config, X_train_full, y_train_full, X_test_final, y_test_final,
                           groups=None, seed=None, registrar=True):
    """
    Treina o modelo final usando a melhor configuração encontrada no Grid Search.
    Usa TODO o conjunto de desenvolvimento (Treino CV completo) para maximizar o aprendizado.

    Args:
        groups: video_id de cada frase do desenvolvimento; na partição por vídeo,
            o dev interno do early stopping é formado por vídeos inteiros.
        seed: semente do treino; por padrão, a da configuração (42).
        registrar: marca o modelo como referência da rodada para as análises a
            jusante (experimento.caminho_modelo_referencia).

    Returns:
        Caminho do modelo salvo, ou None em caso de erro.
    """
    if not best_config:
        logger.warning("Nenhuma melhor configuração encontrada. Pulando treinamento final.")
        return None

    logger.info("\n" + "=" * 70)
    logger.info("TREINANDO MODELO FINAL (BEST CONFIG)")
    logger.info("=" * 70)
    
    config = best_config['config']
    logger.info(f"Usando configuração vencedora:")
    logger.info(f"{config}")

    # Prepara kwargs da melhor configuração
    train_kwargs = {}
    for p_name, p_val in config.items():
        if p_name in GRID_PARAM_MAPPING:
            target_arg = GRID_PARAM_MAPPING[p_name]
            train_kwargs[target_arg] = p_val
            if p_name == 'batch_size':
                train_kwargs['per_device_eval_batch_size'] = p_val

    train_kwargs.setdefault('loss_type', 'focal')
    train_kwargs.setdefault('label_smoothing', 0.1)
    train_kwargs['focal_gamma'] = float(config.get('focal_gamma', 2.0))
    train_kwargs['seed'] = int(seed if seed is not None else config.get('seed', 42))

    X_full = X_train_full.tolist() if isinstance(X_train_full, np.ndarray) else list(X_train_full)
    y_full = y_train_full.tolist() if isinstance(y_train_full, np.ndarray) else list(y_train_full)

    # O dev interno sai do próprio conjunto de desenvolvimento, nunca do teste:
    # é ele que escolhe a época do modelo final, exatamente como nos folds.
    usar_melhor_epoca = bool(config.get('usar_melhor_epoca', True))
    if usar_melhor_epoca:
        X_tr, y_tr, X_dev, y_dev = separar_dev_interno(
            X_full, y_full, random_state=train_kwargs['seed'],
            grupos=None if groups is None else np.asarray(groups)
        )
    else:
        X_tr, y_tr = X_full, y_full
        X_dev, y_dev = X_test_final, y_test_final
    train_kwargs['usar_melhor_epoca'] = usar_melhor_epoca

    if config.get('oversample', True):
        logger.info("Aplicando Oversampling no conjunto de treino final...")
        X_tr, y_tr = apply_oversampling(X_tr, y_tr)
    logger.info(f"Treino final: {len(X_tr)} amostras | dev interno: {len(X_dev)}")

    # Instancia novo modelo
    base_model = config.get('base_model')
    final_model = BertimbauToxicity(config={'base_model': base_model} if base_model else None)

    # Define nome fixo para facilitar carregamento no evaluate
    # O timestamp garante que não sobrescrevemos backups, mas o evaluate busca pelo mais recente
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"TOX_BEST_MODEL_seed{train_kwargs['seed']}_{timestamp}"

    # Treina
    try:
        results = final_model.train_model(
            train_texts=X_tr,
            train_labels=y_tr,
            val_texts=X_dev,
            val_labels=y_dev,
            config_name='default',
            experiment_name=experiment_name,
            save_artifacts=True,
            **train_kwargs
        )

        output_dir = results['model_path']
        logger.info(f"MODELO FINAL TREINADO E SALVO EM: {output_dir}")
        if registrar:
            experimento.registrar_modelo_referencia(output_dir)
            logger.info(f"Registrado como modelo de referência da rodada {experimento.tag()}")
        return output_dir

    except Exception as e:
        logger.error(f"Erro ao treinar modelo final: {e}")
        return None
    finally:
        try:
            import torch
            del final_model
            torch.cuda.empty_cache()
        except Exception:
            pass


# Sementes do treino final. A primeira é a de referência (a mesma da busca):
# é o modelo que as análises a jusante carregam. As demais medem a variação do
# desempenho no teste atribuível só à inicialização.
SEMENTES_FINAIS = [42, 7, 2024, 11, 99]


def predizer_probabilidades(model, textos, batch_size=64):
    """Distribuição softmax sobre as três classes, uma linha por frase."""
    import torch

    model.model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(textos), batch_size):
            entradas = model.tokenizer(
                textos[i:i + batch_size], padding='max_length', truncation=True,
                max_length=model.max_length, return_tensors='pt'
            )
            entradas = {k: v.to(model.device) for k, v in entradas.items()}
            probs.append(torch.softmax(model.model(**entradas).logits, dim=-1).cpu().numpy())
    return np.vstack(probs) if probs else np.zeros((0, 3))


def _metricas_teste(y_true, probs):
    """Macro F1, acurácia e métricas por classe a partir das probabilidades."""
    from sklearn.metrics import classification_report
    preds = probs.argmax(axis=1)
    rel = classification_report(y_true, preds, labels=[0, 1, 2],
                                target_names=['Nenhuma', 'Leve', 'Severa'],
                                output_dict=True, zero_division=0)
    return {
        'f1_macro': float(f1_score(y_true, preds, average='macro', zero_division=0)),
        'accuracy': float(accuracy_score(y_true, preds)),
        'por_classe': {c: {k: float(v) for k, v in rel[c].items()}
                       for c in ('Nenhuma', 'Leve', 'Severa')},
    }


def _melhor_config_salva():
    """Configuração vencedora da busca mais recente desta rodada."""
    pasta = experimento.dir_resultados('toxicity_gridsearch')
    arquivos = sorted(pasta.glob('gridsearch_optimized_search_*.json'))
    if not arquivos:
        raise FileNotFoundError(f"Nenhuma busca concluída em {pasta}; rode --mode search antes.")
    with open(arquivos[-1], encoding='utf-8') as f:
        dados = json.load(f)
    if not dados.get('best_config'):
        raise ValueError(f"{arquivos[-1].name} não tem best_config")
    return dados['best_config'], arquivos[-1].name


def _descartar_pesos(pasta_modelo):
    """
    Apaga os pesos de um modelo final, mantendo config, tokenizer e metadados.

    As sementes que não são a de referência só servem para medir a variação do
    desempenho no teste; suas probabilidades ficam gravadas em CSV, e os pesos
    (~440 MB cada) não são usados por nenhuma análise a jusante.
    """
    pasta_modelo = Path(pasta_modelo)
    liberados = 0
    for nome in ('model.safetensors', 'pytorch_model.bin'):
        arq = pasta_modelo / nome
        if arq.exists():
            liberados += arq.stat().st_size
            arq.unlink()
    (pasta_modelo / 'PESOS_DESCARTADOS.txt').write_text(
        'Pesos apagados após a avaliação no teste (semente não usada nas análises a '
        'jusante). Métricas e probabilidades em results/<tag>/modelo_final/.\n',
        encoding='utf-8')
    logger.info(f"  pesos de {pasta_modelo.name} descartados ({liberados / 1e6:.0f} MB)")


def run_final(sementes=None, manter_modelos=False):
    """
    Treino final com várias sementes e predições out-of-fold da configuração vencedora.

    1. Para cada semente: treina no desenvolvimento inteiro (dev interno agrupado
       por vídeo para o early stopping) e avalia no teste retido. Reporta média e
       desvio-padrão entre sementes -- a variação do teste atribuível só à
       inicialização. A primeira semente é registrada como modelo de referência.
    2. Predições out-of-fold: a configuração vencedora (semente de referência) é
       treinada em cada um dos 5 folds e prediz o fold de validação. As
       probabilidades servem para escolher limiares e calibrar o score sem tocar
       no teste.
    """
    sementes = list(sementes or SEMENTES_FINAIS)
    melhor, origem = _melhor_config_salva()
    config = melhor['config']
    logger.info(f"Rodada {experimento.tag()} | configuração vencedora ({origem}): {config}")

    (X_list, y_list, g_list), (X_test, y_test), (df, train_idx, test_idx) = \
        get_data_for_cv_and_test(test_size=0.20, return_groups=True, return_frame=True)
    X_cv, y_cv, g_cv = np.array(X_list), np.array(y_list), np.array(g_list)
    y_test = np.array(y_test)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saida_dir = experimento.dir_resultados('modelo_final')

    # --- 1. sementes ---------------------------------------------------------
    por_semente, probs_teste = [], {}
    for i, semente in enumerate(sementes):
        caminho = train_final_best_model(melhor, X_cv, y_cv, X_test, y_test,
                                         groups=g_cv, seed=semente, registrar=(i == 0))
        if not caminho:
            continue
        modelo = BertimbauToxicity(model_path=str(caminho))
        probs = predizer_probabilidades(modelo, list(X_test))
        del modelo
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        m = _metricas_teste(y_test, probs)
        m.update({'seed': semente, 'modelo': Path(caminho).name})
        por_semente.append(m)
        probs_teste[semente] = probs
        logger.info(f"  semente {semente}: macro F1 no teste = {m['f1_macro']:.4f}")
        if i > 0 and not manter_modelos:
            _descartar_pesos(caminho)

    # O treino final feito ao fim da busca usa a mesma semente de referência e fica
    # redundante com o modelo registrado acima: descarta os pesos dele também.
    if not manter_modelos and por_semente:
        referencia = experimento.caminho_modelo_referencia()
        for pasta in experimento.dir_modelos().glob('TOX_BEST_MODEL_*'):
            if pasta.resolve() != referencia.resolve() and (pasta / 'config.json').exists():
                _descartar_pesos(pasta)

    f1s = [m['f1_macro'] for m in por_semente]
    resumo_sementes = {
        'f1_macro_media': float(np.mean(f1s)) if f1s else None,
        'f1_macro_dp': float(np.std(f1s, ddof=1)) if len(f1s) > 1 else None,
        'f1_macro_min': float(np.min(f1s)) if f1s else None,
        'f1_macro_max': float(np.max(f1s)) if f1s else None,
    }

    # --- 2. out-of-fold ------------------------------------------------------
    oof = np.full((len(X_cv), 3), np.nan)
    fold_de = np.zeros(len(X_cv), dtype=int)
    for fold, (tr, va) in enumerate(get_kfold_split(X_cv, y_cv, n_splits=K_FOLDS,
                                                   groups=g_cv), 1):
        cfg = dict(config, seed=sementes[0])
        X_tr, y_tr, X_dev, y_dev = separar_dev_interno(
            X_cv[tr].tolist(), y_cv[tr].tolist(), random_state=sementes[0], grupos=g_cv[tr]
        )
        if cfg.get('oversample', True):
            X_tr, y_tr = apply_oversampling(X_tr, y_tr)
        kwargs = {GRID_PARAM_MAPPING[k]: v for k, v in cfg.items() if k in GRID_PARAM_MAPPING}
        if 'batch_size' in cfg:
            kwargs['per_device_eval_batch_size'] = cfg['batch_size']
        kwargs.setdefault('loss_type', 'focal')
        kwargs.setdefault('label_smoothing', 0.1)
        kwargs['focal_gamma'] = float(cfg.get('focal_gamma', 2.0))
        kwargs['usar_melhor_epoca'] = True
        kwargs['seed'] = sementes[0]
        modelo = BertimbauToxicity()
        modelo.train_model(train_texts=X_tr, train_labels=y_tr, val_texts=X_dev,
                           val_labels=y_dev, config_name='default',
                           experiment_name=f'TOX_oof_fold_{fold}', save_artifacts=False,
                           **kwargs)
        oof[va] = predizer_probabilidades(modelo, X_cv[va].tolist())
        fold_de[va] = fold
        del modelo
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        f1 = f1_score(y_cv[va], oof[va].argmax(axis=1), average='macro', zero_division=0)
        logger.info(f"  OOF fold {fold}: macro F1 = {f1:.4f}")

    # --- persistência --------------------------------------------------------
    ids_dev = df['id_sentenca'].values[train_idx]
    ids_teste = df['id_sentenca'].values[test_idx]
    linhas_oof = pd.DataFrame({
        'id_sentenca': ids_dev, 'video_id': g_cv, 'fold': fold_de, 'rotulo': y_cv,
        'p_nenhuma': oof[:, 0], 'p_leve': oof[:, 1], 'p_severa': oof[:, 2],
    })
    linhas_oof.to_csv(saida_dir / f'probabilidades_oof_{timestamp}.csv',
                      sep=';', index=False, encoding='utf-8')
    linhas_teste = pd.DataFrame({'id_sentenca': ids_teste, 'rotulo': y_test})
    for semente, probs in probs_teste.items():
        for j, nome in enumerate(('p_nenhuma', 'p_leve', 'p_severa')):
            linhas_teste[f'{nome}_s{semente}'] = probs[:, j]
    linhas_teste.to_csv(saida_dir / f'probabilidades_teste_{timestamp}.csv',
                        sep=';', index=False, encoding='utf-8')

    validos = ~np.isnan(oof).any(axis=1)
    resumo = {
        'timestamp': timestamp,
        'experimento': experimento.descricao(),
        'busca_origem': origem,
        'config': config,
        'sementes': sementes,
        'semente_referencia': sementes[0],
        'teste_por_semente': por_semente,
        'teste_resumo': resumo_sementes,
        'oof_f1_macro': float(f1_score(y_cv[validos], oof[validos].argmax(axis=1),
                                       average='macro', zero_division=0)),
        'n_dev': int(len(X_cv)), 'n_teste': int(len(X_test)),
    }
    with open(saida_dir / f'modelo_final_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nRodada {experimento.tag()}")
    if f1s:
        print(f"Teste (macro F1) em {len(f1s)} sementes: "
              f"{resumo_sementes['f1_macro_media']:.4f} ± {resumo_sementes['f1_macro_dp'] or 0:.4f}")
    print(f"OOF (macro F1): {resumo['oof_f1_macro']:.4f}")
    print(f"Salvo em: {saida_dir}")
    return resumo


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Grid Search para Toxicidade')
    parser.add_argument('--mode', choices=['fast', 'full', 'optimized', 'search', 'final'],
                        default='fast',
                        help='Modo de execução: fast (1 config), full (72 configs), '
                             'optimized (successive halving em 2 estágios), '
                             'search (busca aleatória no espaço ampliado + halving '
                             'em 3 estágios, recomendado), final (treino final com '
                             'várias sementes + predições out-of-fold da vencedora)')
    parser.add_argument('--preprocess', action='store_true',
                        help='Aplica pré-processamento (limpeza + lematização) aos textos')
    parser.add_argument('--sementes', type=int, nargs='+', default=None,
                        help=f'Sementes do modo final (padrão: {SEMENTES_FINAIS})')
    parser.add_argument('--manter-modelos', action='store_true',
                        help='modo final: mantém os pesos de todas as sementes '
                             '(por padrão só o modelo de referência fica com pesos)')
    args = parser.parse_args()

    logger.info(f"Rodada experimental: {experimento.descricao()}")
    if args.mode == 'final':
        run_final(args.sementes, manter_modelos=args.manter_modelos)
        return
    all_results, best_config, best_model_path = run_grid_search(mode=args.mode, preprocess=args.preprocess)


if __name__ == "__main__":
    main()
