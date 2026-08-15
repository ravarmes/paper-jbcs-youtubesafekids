"""
Baselines com LLMs generativos para Detecção de Toxicidade (Revisão — item C1).
-------------------------------------------------------------------------------
Atende à crítica (levantada pelos 3 revisores) de que o artigo não considera
modelos generativos locais e abertos (Llama, Gemma, Mistral...), que a
literatura mostra classificarem de forma comparável ou superior a transformers.

Avalia 2 (ou mais) LLMs em modo *few-shot* no MESMO conjunto de teste retido
(20%, random_state=42) usado pelos baselines clássicos e pelo BERTimbau,
garantindo comparabilidade direta na coluna "F1 (Teste)".

Como os LLMs operam em *zero/few-shot* (sem fine-tuning), NÃO há validação
cruzada: a coluna "F1 (CV)" não se aplica a eles e é reportada como "—".

Backend: Ollama (https://ollama.com) — modelos locais e abertos, sem nuvem.
  Instalar o Ollama e baixar os modelos ANTES de rodar, por exemplo:
      ollama pull llama3.1:8b
      ollama pull gemma2:9b
  Usa apenas a biblioteca padrão (urllib) — nenhuma dependência pip nova.

Uso:
    # sobe o servidor do Ollama (em outro terminal): ollama serve
    python src/app/nlp/training/train_toxicity_llm_baselines.py
    python src/app/nlp/training/train_toxicity_llm_baselines.py --limit 60
    python src/app/nlp/training/train_toxicity_llm_baselines.py --models llama3.1:8b mistral:7b

Config por variáveis de ambiente:
    OLLAMA_HOST   (default http://localhost:11434)
    TOX_LLM_MODELS (lista separada por vírgula; sobrepõe o default)
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import urllib.request
import urllib.error
from datetime import datetime

import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

# Setup do path do projeto (src/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp.datasets.prepare_data_toxicity import get_data_for_cv_and_test
from app.nlp.config import PATHS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ======================================================================
# Configuração
# ======================================================================

RANDOM_STATE = 42
CLASS_NAMES = ['Nenhuma', 'Leve', 'Severa']

OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

# 2 LLMs abertos por padrão (ajuste conforme o que você baixou no Ollama).
# Os revisores citaram explicitamente llama / gemma / mistral.
DEFAULT_MODELS = ['llama3.1:8b', 'gemma2:9b']

# Nº de exemplos few-shot por classe (amostrados do TREINO, nunca do teste).
FEWSHOT_PER_CLASS = 2

SYSTEM_PROMPT = (
    "Você é um sistema de moderação de conteúdo em português "
    "brasileiro. Sua tarefa é classificar o grau de toxicidade de uma frase "
    "extraída da transcrição de um vídeo, em três níveis:\n"
    "0 = Nenhuma: texto apropriado, isento de linguagem agressiva.\n"
    "1 = Leve: ofensividade indireta — críticas dirigidas a objetos, situações "
    "ou entidades abstratas, sem alvo pessoal (ex.: \"Que porra de ônibus\").\n"
    "2 = Severa: ofensividade direta — insultos, ameaças ou discurso de ódio "
    "dirigidos explicitamente a uma pessoa ou grupo (ex.: \"Este idiota do "
    "Robson fez isso\").\n"
    "Responda APENAS com o número 0, 1 ou 2, sem nenhuma outra palavra."
)


def montar_fewshot(X_train, y_train):
    """Seleciona exemplos few-shot balanceados por classe, de forma determinística."""
    rng = random.Random(RANDOM_STATE)
    por_classe = {0: [], 1: [], 2: []}
    for txt, lab in zip(X_train, y_train):
        por_classe[int(lab)].append(txt)
    exemplos = []
    for cls in (0, 1, 2):
        pool = por_classe[cls]
        rng.shuffle(pool)
        for txt in pool[:FEWSHOT_PER_CLASS]:
            exemplos.append((txt, cls))
    rng.shuffle(exemplos)
    return exemplos


def montar_mensagens(fewshot, frase):
    """Monta o array de mensagens (system + few-shot + a frase alvo)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for txt, cls in fewshot:
        messages.append({"role": "user", "content": f"Frase: {txt}"})
        messages.append({"role": "assistant", "content": str(cls)})
    messages.append({"role": "user", "content": f"Frase: {frase}"})
    return messages


def parse_label(resposta: str) -> int:
    """Extrai 0/1/2 da resposta do LLM, com fallback por palavra-chave."""
    if resposta is None:
        return 0
    for ch in resposta:
        if ch in '012':
            return int(ch)
    baixo = resposta.lower()
    if 'severa' in baixo:
        return 2
    if 'leve' in baixo:
        return 1
    if 'nenhuma' in baixo:
        return 0
    return 0  # fallback conservador: assume Nenhuma se não parsear


def chamar_ollama(model: str, messages, timeout: int = 120, retries: int = 3) -> str:
    """Chama o endpoint /api/chat do Ollama (stream=False) e devolve o conteúdo."""
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 5},
    }
    data = json.dumps(payload).encode('utf-8')
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, data=data,
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode('utf-8'))
            return body.get('message', {}).get('content', '')
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
            logger.warning(f"  tentativa {attempt}/{retries} falhou: {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Falha ao chamar Ollama ({model}): {last_err}")


def referencia_bertimbau():
    """
    Lê o desempenho do modelo ajustado a partir do resultado de avaliação mais
    recente, em vez de repetir constantes no código — que envelhecem a cada
    retreino e passam a mentir silenciosamente.
    """
    import glob
    padrao = os.path.join(PATHS['evaluation_dir'], 'toxicity_evaluation',
                          'evaluation_metrics_TOX_*.json')
    arquivos = glob.glob(padrao)
    if not arquivos:
        return None
    with open(max(arquivos, key=os.path.getmtime), encoding='utf-8') as f:
        d = json.load(f)
    return {'test_f1': d['f1_macro'], 'test_accuracy': d['accuracy']}


def avaliar_modelo(model, fewshot, X_test, y_test, limit=None, idx_sem_analogo=None):
    """Classifica o conjunto de teste com um LLM e calcula as métricas."""
    if limit:
        X_test, y_test = X_test[:limit], y_test[:limit]

    y_true, y_pred = [], []
    n = len(X_test)
    t0 = time.time()
    for i, (frase, gold) in enumerate(zip(X_test, y_test), 1):
        messages = montar_mensagens(fewshot, frase)
        resposta = chamar_ollama(model, messages)
        pred = parse_label(resposta)
        y_true.append(int(gold))
        y_pred.append(pred)
        if i % 25 == 0 or i == n:
            logger.info(f"  [{model}] {i}/{n} frases classificadas...")

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    metrics = {
        'f1': float(f1_score(y_true, y_pred, average='macro')),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, zero_division=0),
        'n_amostras': int(n),
        'segundos': round(time.time() - t0, 1),
    }

    # Mesmo recorte usado pelos demais modelos: as frases de teste que não têm
    # análogo próximo no treino, onde se mede generalização de fato.
    if idx_sem_analogo:
        sel = [i for i in idx_sem_analogo if i < len(y_true)]
        if sel:
            metrics['por_similaridade'] = {
                'sem_analogo': {
                    'n': len(sel),
                    'f1': float(f1_score(y_true[sel], y_pred[sel],
                                         average='macro', zero_division=0)),
                    'accuracy': float(accuracy_score(y_true[sel], y_pred[sel])),
                }
            }

    return metrics


def print_tabela(resultados):
    print("\n" + "=" * 80)
    print("BASELINES LLM — DETECÇÃO DE TOXICIDADE (few-shot, conjunto de teste)")
    print("=" * 80)
    print(f"{'Modelo':<22} {'F1 (CV)':<10} {'F1 (Teste)':<12} {'Acc':<8} {'Prec':<8} {'Rec':<8}")
    print("-" * 80)
    for name, m in sorted(resultados.items(), key=lambda x: x[1]['f1'], reverse=True):
        print(f"{name:<22} {'—':<10} {m['f1']:<12.4f} {m['accuracy']:<8.4f} "
              f"{m['precision']:<8.4f} {m['recall']:<8.4f}")
    print("-" * 80)
    ref = referencia_bertimbau()
    if ref:
        print(f"{'BERTimbau (ref.)':<22} {'—':<10} {ref['test_f1']:<12.4f} "
              f"{ref['test_accuracy']:<8.4f}")
    print("=" * 80)
    print("Obs.: LLMs são zero/few-shot (sem fine-tuning) — coluna F1 (CV) não se aplica.")


def salvar(resultados, models, limit):
    out_dir = PATHS.get('evaluation_dir', os.path.join(project_root, 'results'))
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"llm_baselines_toxicity_{ts}.json")
    saida = {
        'timestamp': ts,
        'backend': 'ollama',
        'ollama_host': OLLAMA_HOST,
        'models': models,
        'methodology': {
            'mode': 'few-shot',
            'fewshot_per_class': FEWSHOT_PER_CLASS,
            'test_size': 0.20,
            'random_state': RANDOM_STATE,
            'limit': limit,
        },
        'results': resultados,
        'bertimbau_reference': referencia_bertimbau(),
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResultados salvos em: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Baselines LLM para toxicidade (Ollama)")
    parser.add_argument('--models', nargs='*', default=None,
                        help="Modelos Ollama (default: llama3.1:8b gemma2:9b)")
    parser.add_argument('--limit', type=int, default=None,
                        help="Avalia só as N primeiras frases de teste (ensaio rápido)")
    args = parser.parse_args()

    env_models = os.getenv('TOX_LLM_MODELS')
    models = args.models or (env_models.split(',') if env_models else DEFAULT_MODELS)
    models = [m.strip() for m in models if m.strip()]

    print("=" * 80)
    print("BASELINES LLM — DETECÇÃO DE TOXICIDADE")
    print(f"Backend Ollama em {OLLAMA_HOST} | Modelos: {', '.join(models)}")
    print("=" * 80)

    (X_train_cv, y_train_cv), (X_test, y_test) = get_data_for_cv_and_test(test_size=0.20)
    logger.info(f"Teste: {len(X_test)} amostras | Few-shot: {FEWSHOT_PER_CLASS}/classe")

    fewshot = montar_fewshot(X_train_cv, y_train_cv)

    # Mesmo recorte de proximidade usado pelo BERTimbau e pelas baselines
    from app.nlp.evaluation.evaluate_toxicity import (
        separar_por_similaridade, LIMIAR_SIMILARIDADE
    )
    _, idx_sem_analogo, _ = separar_por_similaridade(
        list(X_train_cv), list(X_test), limiar=LIMIAR_SIMILARIDADE
    )
    logger.info(f"Frases de teste sem analogo proximo: {len(idx_sem_analogo)}")

    resultados = {}
    for model in models:
        logger.info(f"\n{'='*60}\nAvaliando LLM: {model}\n{'='*60}")
        try:
            resultados[model] = avaliar_modelo(model, fewshot, X_test, y_test,
                                               args.limit, idx_sem_analogo)
            m = resultados[model]
            logger.info(f"  {model}: F1={m['f1']:.4f} | Acc={m['accuracy']:.4f} "
                        f"({m['segundos']}s)")
        except Exception as e:
            logger.error(f"  Falha ao avaliar {model}: {e}")

    if resultados:
        print_tabela(resultados)
        salvar(resultados, models, args.limit)
    else:
        print("Nenhum modelo avaliado com sucesso. Verifique se o 'ollama serve' está ativo "
              "e se os modelos foram baixados (ollama pull ...).")


if __name__ == "__main__":
    main()
