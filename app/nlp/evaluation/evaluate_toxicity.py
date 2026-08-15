"""
Script de avaliação final para Detecção de Toxicidade (TOX).

Este script avalia o modelo treinado no conjunto de teste holdout (20%)
e gera relatórios detalhados com métricas e matriz de confusão.
"""

import logging
import sys
import os
import pandas as pd
from pathlib import Path

# Adiciona o diretório raiz ao path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp.models.bertimbau_toxicity import BertimbauToxicity
from app.nlp.evaluation.model_evaluator import ModelEvaluator
from app.nlp.datasets.prepare_data_toxicity import get_data_for_cv_and_test
from app.nlp.utils.preprocessing import preprocess_batch
from app.nlp.config import get_task_config

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Acima deste cosseno (n-gramas de caracteres), considera-se que a frase de
# teste tem um análogo próximo no treino.
LIMIAR_SIMILARIDADE = 0.8


def separar_por_similaridade(train_texts, test_texts, limiar=0.8):
    """
    Divide o teste entre frases com e sem análogo próximo no treino.

    Mesmo sem repetições exatas, o corpus tem frases que diferem de outra do
    treino por uma ou duas palavras. Modelos lexicais acertam essas quase por
    construção, então a média global mistura duas dificuldades muito
    diferentes. A similaridade é o cosseno sobre n-gramas de caracteres (3 a 5),
    que captura variação morfológica sem depender de tokenização.

    Args:
        train_texts: Frases do conjunto de treino
        test_texts: Frases do conjunto de teste
        limiar: acima disso, considera-se que há análogo próximo

    Returns:
        (indices_com_analogo, indices_sem_analogo, similaridade_maxima_por_frase)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    vetorizador = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=1)
    vetorizador.fit(train_texts)

    similaridades = cosine_similarity(
        vetorizador.transform(test_texts), vetorizador.transform(train_texts)
    ).max(axis=1)

    com = [i for i, s in enumerate(similaridades) if s > limiar]
    sem = [i for i, s in enumerate(similaridades) if s <= limiar]
    return com, sem, similaridades.tolist()


def avaliar_subconjunto(model, evaluator, textos, rotulos):
    """Avalia o modelo num subconjunto e devolve só as métricas agregadas."""
    if not textos:
        return None
    m = evaluator.evaluate_model(
        model=model.model, tokenizer=model.tokenizer,
        test_texts=textos, test_labels=rotulos,
        max_length=model.max_length, batch_size=32
    )
    return {
        'n': len(textos),
        'accuracy': m['accuracy'],
        'f1_macro': m['f1_macro'],
        'f1_per_class': m['f1_per_class'],
        'support_per_class': m['support_per_class'],
    }


def avaliar_sentencas_distintas(model, evaluator, test_texts, test_labels):
    """
    Reavalia o conjunto de teste contando cada texto distinto uma única vez.

    O corpus contém frases-molde repetidas; na métrica por ocorrência, uma
    frase que aparece 30 vezes pesa 30 vezes. Esta avaliação paralela mostra
    o desempenho quando cada sentença vale um voto — se os dois números forem
    próximos, o resultado não vem da repetição.

    Args:
        model: BertimbauToxicity já carregado
        evaluator: ModelEvaluator configurado para a tarefa
        test_texts: Textos do conjunto de teste
        test_labels: Rótulos correspondentes

    Returns:
        Dict com as métricas agregadas sobre os textos distintos.
    """
    vistos = {}
    for texto, rotulo in zip(test_texts, test_labels):
        chave = texto.strip().lower()
        if chave not in vistos:
            vistos[chave] = (texto, rotulo)

    textos = [t for t, _ in vistos.values()]
    rotulos = [r for _, r in vistos.values()]

    logger.info(f"Reavaliando sobre {len(textos)} textos distintos "
                f"(de {len(test_texts)} ocorrências)...")

    metricas = evaluator.evaluate_model(
        model=model.model,
        tokenizer=model.tokenizer,
        test_texts=textos,
        test_labels=rotulos,
        max_length=model.max_length,
        batch_size=32
    )

    return {
        'n_ocorrencias': len(test_texts),
        'n_textos_distintos': len(textos),
        'accuracy': metricas['accuracy'],
        'f1_macro': metricas['f1_macro'],
        'precision_macro': metricas['precision_macro'],
        'recall_macro': metricas['recall_macro'],
        'f1_per_class': metricas['f1_per_class'],
        'support_per_class': metricas['support_per_class'],
        'confusion_matrix': metricas['confusion_matrix'],
    }


def main():
    """Função principal de avaliação."""
    import argparse
    parser = argparse.ArgumentParser(description='Avaliação de Toxicidade')
    parser.add_argument('--preprocess', action='store_true',
                        help='Aplica pré-processamento (limpeza + lematização) se o modelo foi treinado assim')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Avaliação do Modelo de Detecção de Toxicidade")
    if args.preprocess:
        logger.info("PRÉ-PROCESSAMENTO: ATIVADO (Limpeza + Lematização)")
    logger.info("=" * 60)
    
    # Carrega treino e teste: o treino é necessário para medir quais frases de
    # teste têm análogo próximo entre as vistas pelo modelo.
    logger.info("Carregando conjunto de TESTE (Holdout 20%)...")
    (train_texts, _), (test_texts, test_labels) = get_data_for_cv_and_test()
    
    logger.info(f"Total de amostras de teste: {len(test_texts)}")
    logger.info(f"Distribuição: {pd.Series(test_labels).value_counts().sort_index().to_dict()}")
    
    # Aplica pré-processamento se solicitado
    if args.preprocess:
        logger.info("Aplicando pré-processamento aos dados de teste...")
        test_texts = preprocess_batch(test_texts, apply_cleaning=True, apply_lemmatization=True)
    
    # Configuração da tarefa
    task_config = get_task_config('TOX')
    class_names = task_config['classes']
    
    # Caminho do modelo treinado (busca o mais recente)
    models_dir = Path(__file__).parent.parent / 'models' / 'trained'
    
    # Lista modelos de toxicidade disponíveis.
    # O grid search deixa uma pasta TOX_* por fold contendo apenas logs; só
    # interessam as que têm de fato um modelo salvo (config.json).
    model_paths = [p for p in models_dir.glob('TOX_*') if (p / 'config.json').exists()]
    if not model_paths:
        logger.error(f"Nenhum modelo TOX_* com pesos salvos encontrado em {models_dir}")
        logger.error("Por favor, treine o modelo primeiro usando train_toxicity_gridsearch.py")
        return
    
    # Usa o modelo mais recente
    model_path = max(model_paths, key=lambda p: p.stat().st_mtime)
    logger.info(f"Carregando modelo de: {model_path}")
    
    # Carrega modelo treinado
    model = BertimbauToxicity(model_path=str(model_path))
    
    # Cria avaliador
    output_dir = Path(__file__).parent / 'results' / 'toxicity_evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = ModelEvaluator(
        task_name='TOX',
        class_names=class_names,
        output_dir=str(output_dir)
    )
    
    # Avalia modelo
    logger.info("Avaliando modelo no conjunto de teste...")
    results = evaluator.evaluate_model(
        model=model.model,
        tokenizer=model.tokenizer,
        test_texts=test_texts,
        test_labels=test_labels,
        max_length=model.max_length,
        batch_size=32
    )

    # Robustez 1: se houver frases repetidas, uma frase repetida N vezes pesa N
    # vezes na métrica. Reavalia contando cada texto distinto uma única vez.
    results['por_sentenca_distinta'] = avaliar_sentencas_distintas(
        model, evaluator, test_texts, test_labels
    )

    # Robustez 2: separa o teste entre frases com e sem análogo próximo no
    # treino. É o segundo grupo que mede generalização para formulações novas.
    com, sem, similaridades = separar_por_similaridade(train_texts, test_texts, limiar=LIMIAR_SIMILARIDADE)
    logger.info(f"Teste: {len(com)} frases com análogo próximo (sim > {LIMIAR_SIMILARIDADE}) "
                f"e {len(sem)} sem análogo")
    results['por_similaridade'] = {
        'limiar': LIMIAR_SIMILARIDADE,
        'similaridade_maxima': similaridades,
        'com_analogo': avaliar_subconjunto(
            model, evaluator, [test_texts[i] for i in com], [test_labels[i] for i in com]
        ),
        'sem_analogo': avaliar_subconjunto(
            model, evaluator, [test_texts[i] for i in sem], [test_labels[i] for i in sem]
        ),
    }

    # Gera relatório e gráficos
    try:
        logger.info("Gerando relatório de avaliação e matriz de confusão...")
        evaluator.generate_report(results, model_path=str(model_path), save_plots=True)
    except Exception as e:
        logger.error(f"Erro ao gerar relatório: {e}")
    
    # Imprime resultados
    logger.info("=" * 60)
    logger.info("Avaliação concluída!")
    logger.info("=" * 60)
    
    print("\n" + "=" * 60)
    print("RESULTADOS DA AVALIAÇÃO (CONJUNTO HOLDOUT - 20%)")
    print("=" * 60)
    print(f"Acurácia: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"F1-Score (macro): {results['f1_macro']:.4f}")
    print(f"F1-Score (weighted): {results['f1_weighted']:.4f}")
    print(f"Precisão (macro): {results['precision_macro']:.4f}")
    print(f"Recall (macro): {results['recall_macro']:.4f}")
    print("=" * 60)

    distintas = results['por_sentenca_distinta']
    print(f"\nSobre textos DISTINTOS ({distintas['n_textos_distintos']} de "
          f"{distintas['n_ocorrencias']} ocorrências):")
    print(f"  Acurácia: {distintas['accuracy']:.4f} | "
          f"F1 (macro): {distintas['f1_macro']:.4f}")

    sim = results['por_similaridade']
    print(f"\nPor proximidade com o treino (limiar {sim['limiar']}):")
    for chave, rotulo in (('com_analogo', 'COM análogo próximo'),
                          ('sem_analogo', 'SEM análogo próximo')):
        bloco = sim[chave]
        if bloco:
            print(f"  {rotulo} (n={bloco['n']}): F1 (macro) = {bloco['f1_macro']:.4f} | "
                  f"Acurácia = {bloco['accuracy']:.4f}")
    print("=" * 60)
    
    if 'classification_report' in results:
        print("\nRelatório de Classificação por Classe:")
        report_dict = results['classification_report']
        print(pd.DataFrame(report_dict).transpose())
    
    print(f"\nResultados salvos em: {output_dir}")
    print(f"Matriz de Confusão salva em: {output_dir}")
    
    # Informações para o artigo
    print("\n" + "=" * 60)
    print("DADOS PARA O ARTIGO CIENTÍFICO")
    print("=" * 60)
    print(f"Conjunto de Teste: {len(test_texts)} amostras")
    print(f"Classes: {class_names}")
    print(f"Modelo: BERTimbau Base (neuralmind/bert-base-portuguese-cased)")
    print(f"Caminho do modelo: {model_path}")


if __name__ == "__main__":
    main()
