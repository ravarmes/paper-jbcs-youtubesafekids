"""
Referências SEM ajuste fino no corpus — quanto do desempenho vem do fine-tuning?

O BERTimbau puro não classifica toxicidade: é um modelo de linguagem mascarada,
sem cabeça de classificação para a tarefa. "Zero-shot" com ele carregado e uma
cabeça nova daria acerto aleatório e não mediria nada. Este script implementa as
três referências que respondem à pergunta de forma interpretável:

1. SONDA LINEAR (`--modo sonda`)
   Representações congeladas do BERTimbau (média dos estados ocultos da última
   camada) alimentando uma regressão logística. Mesmos folds e mesmo teste do
   ajuste fino. Como a representação é idêntica e só o ajuste fino é removido,
   a diferença isola exatamente o ganho do fine-tuning.

2. ZERO-SHOT POR MÁSCARA (`--modo cloze`)
   Classificação por preenchimento de lacuna: monta-se "<frase> Isso é uma fala
   [MASK]." e comparam-se as probabilidades que o modelo de linguagem atribui a
   palavras-âncora de cada classe. Mede o que o BERTimbau já sabe sobre
   ofensividade sem nenhuma supervisão. É sensível ao molde e às âncoras
   escolhidas, então roda com vários moldes e reporta a dispersão.

3. MODELOS PRONTOS DE TOXICIDADE (`--modo prontos`)
   Dois classificadores públicos em português, ambos derivados do BERTimbau,
   compostos no esquema de três classes deste trabalho:
     - `ruanchaves/bert-base-portuguese-cased-hatebr`: ofensivo (sim/não)
     - `dougtrajano/toxicity-target-classification`: UNTARGETED / TARGETED INSULT
   Composição: não ofensivo -> Nenhuma; ofensivo e sem alvo -> Leve;
   ofensivo e com alvo -> Severa. O mapeamento segue a própria definição dos
   rótulos deste corpus (Leve = ofensividade indireta; Severa = dirigida).
   Mede o que se obteria sem construir corpus algum.

Todas as referências são avaliadas no MESMO conjunto de teste retido do ajuste
fino. A sonda linear, por ser treinada nos nossos folds, também reporta
validação cruzada; as demais não têm treino sobre este corpus e por isso não
têm coluna de CV.

Uso:
    python -m app.nlp.training.train_toxicity_sem_finetuning              # tudo
    python -m app.nlp.training.train_toxicity_sem_finetuning --modo sonda
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp.datasets.prepare_data_toxicity import get_data_for_cv_and_test
from app.nlp.utils.data_utils import get_kfold_split
from app.nlp.config import MODEL_CONFIG, PATHS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLASSES = ['Nenhuma', 'Leve', 'Severa']
K_FOLDS = 5
RANDOM_STATE = 42
MODELOS_PRONTOS_DIR = os.path.join(project_root, 'data', 'models')

# Moldes e âncoras do zero-shot por máscara. Vários moldes porque a escolha de
# um único seria arbitrária e o resultado depende bastante dela.
MOLDES_CLOZE = [
    '{frase} Isso é uma fala [MASK].',
    '{frase} Essa frase é [MASK].',
    'Comentário: {frase} Classificação: [MASK].',
]

# Palavras-âncora por classe (a de maior probabilidade define a predição).
# Só entram palavras que o WordPiece do BERTimbau representa como token único —
# subpalavras não têm probabilidade direta na posição da máscara. Candidatas
# naturais como "grosseira", "agressiva" e "educada" ficaram de fora por isso.
ANCORAS = {
    0: ['normal', 'comum', 'simples', 'boa', 'positiva', 'legal'],
    1: ['vulgar', 'pesada', 'baixa', 'ruim'],
    2: ['ofensiva', 'violenta', 'cruel', 'terrível', 'brutal'],
}


def _dispositivo():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _metricas(y_true, y_pred, nome):
    return {
        'modelo': nome,
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_per_class': f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }


# ---------------------------------------------------------------------------
# 1. Sonda linear sobre representações congeladas
# ---------------------------------------------------------------------------

def extrair_embeddings(textos, batch_size=64):
    """
    Média dos estados ocultos da última camada, ponderada pela máscara de
    atenção (ignora o preenchimento). O modelo fica em modo de avaliação e sem
    gradiente: nada aqui é treinado.
    """
    from transformers import BertModel, BertTokenizer

    nome = MODEL_CONFIG['base_model']
    cache = MODEL_CONFIG.get('cache_dir')
    tokenizador = BertTokenizer.from_pretrained(nome, cache_dir=cache)
    modelo = BertModel.from_pretrained(nome, cache_dir=cache).to(_dispositivo()).eval()

    saidas = []
    with torch.no_grad():
        for i in range(0, len(textos), batch_size):
            lote = textos[i:i + batch_size]
            entradas = tokenizador(lote, padding=True, truncation=True,
                                   max_length=MODEL_CONFIG['max_length'],
                                   return_tensors='pt')
            entradas = {k: v.to(_dispositivo()) for k, v in entradas.items()}
            estados = modelo(**entradas).last_hidden_state
            mascara = entradas['attention_mask'].unsqueeze(-1).float()
            media = (estados * mascara).sum(1) / mascara.sum(1).clamp(min=1e-9)
            saidas.append(media.cpu().numpy())

    del modelo
    torch.cuda.empty_cache()
    return np.vstack(saidas)


def rodar_sonda(X_train, y_train, X_test, y_test):
    logger.info("Extraindo representações congeladas do BERTimbau...")
    E_train = extrair_embeddings(list(X_train))
    E_test = extrair_embeddings(list(X_test))
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    def classificador():
        return LogisticRegression(max_iter=3000, class_weight='balanced',
                                  random_state=RANDOM_STATE, multi_class='multinomial')

    f1s = []
    for fold, (tr, va) in enumerate(
        get_kfold_split(np.array(X_train), y_train, n_splits=K_FOLDS,
                        random_state=RANDOM_STATE), 1
    ):
        modelo = classificador().fit(E_train[tr], y_train[tr])
        f1 = f1_score(y_train[va], modelo.predict(E_train[va]), average='macro', zero_division=0)
        f1s.append(f1)
        logger.info(f"  Fold {fold}/{K_FOLDS}: F1={f1:.4f}")

    modelo = classificador().fit(E_train, y_train)
    resultado = _metricas(y_test, modelo.predict(E_test), 'Sonda linear (BERTimbau congelado)')
    resultado['cv_f1_mean'] = float(np.mean(f1s))
    resultado['cv_f1_std'] = float(np.std(f1s))
    logger.info(f"  CV: {resultado['cv_f1_mean']:.4f} ±{resultado['cv_f1_std']:.4f} | "
                f"Teste: {resultado['f1_macro']:.4f}")
    return resultado


# ---------------------------------------------------------------------------
# 2. Zero-shot por preenchimento de máscara
# ---------------------------------------------------------------------------

def rodar_cloze(X_test, y_test):
    from transformers import BertForMaskedLM, BertTokenizer

    nome = MODEL_CONFIG['base_model']
    cache = MODEL_CONFIG.get('cache_dir')
    tokenizador = BertTokenizer.from_pretrained(nome, cache_dir=cache)
    modelo = BertForMaskedLM.from_pretrained(nome, cache_dir=cache).to(_dispositivo()).eval()

    # Âncoras precisam ser um único token no vocabulário para ter probabilidade
    # direta na posição da máscara; as demais são descartadas com aviso.
    ancoras_ids = {}
    for classe, palavras in ANCORAS.items():
        ids = []
        for p in palavras:
            tok = tokenizador.tokenize(p)
            if len(tok) == 1:
                ids.append(tokenizador.convert_tokens_to_ids(tok)[0])
            else:
                logger.warning(f"Âncora '{p}' não é token único ({tok}); descartada")
        ancoras_ids[classe] = ids
        if not ids:
            logger.error(f"Nenhuma âncora válida para a classe {CLASSES[classe]}")

    resultados = []
    for molde in MOLDES_CLOZE:
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_test), 32):
                textos = [molde.format(frase=t) for t in X_test[i:i + 32]]
                entradas = tokenizador(textos, padding=True, truncation=True,
                                       max_length=MODEL_CONFIG['max_length'],
                                       return_tensors='pt')
                entradas = {k: v.to(_dispositivo()) for k, v in entradas.items()}
                logits = modelo(**entradas).logits
                pos = (entradas['input_ids'] == tokenizador.mask_token_id)
                for linha in range(logits.size(0)):
                    j = pos[linha].nonzero()
                    if len(j) == 0:            # máscara truncada: cai na majoritária
                        preds.append(0)
                        continue
                    probs = torch.softmax(logits[linha, j[0, 0]], dim=-1)
                    pontos = [float(probs[ids].max()) if ids else -1.0
                              for _, ids in sorted(ancoras_ids.items())]
                    preds.append(int(np.argmax(pontos)))

        r = _metricas(y_test, preds, f'Zero-shot máscara: "{molde}"')
        r['molde'] = molde
        resultados.append(r)
        logger.info(f"  {molde[:40]:42} F1={r['f1_macro']:.4f}")

    del modelo
    torch.cuda.empty_cache()
    return resultados


# ---------------------------------------------------------------------------
# 3. Modelos prontos de toxicidade em português
# ---------------------------------------------------------------------------

def _prever_binario(caminho, textos, batch_size=32):
    """Roda um classificador binário local e devolve a classe predita."""
    from transformers import AutoTokenizer, BertForSequenceClassification

    tokenizador = AutoTokenizer.from_pretrained(caminho)
    modelo = BertForSequenceClassification.from_pretrained(caminho).to(_dispositivo()).eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(textos), batch_size):
            entradas = tokenizador(textos[i:i + batch_size], padding=True, truncation=True,
                                   max_length=128, return_tensors='pt')
            entradas = {k: v.to(_dispositivo()) for k, v in entradas.items()}
            preds.extend(torch.argmax(modelo(**entradas).logits, dim=-1).cpu().numpy().tolist())

    del modelo
    torch.cuda.empty_cache()
    return preds


def rodar_prontos(X_test, y_test):
    hatebr = os.path.join(MODELOS_PRONTOS_DIR, 'ruanchaves_bert_base_hatebr')
    alvo = os.path.join(MODELOS_PRONTOS_DIR, 'dougtrajano_toxicity_target_classification')

    faltando = [p for p in (hatebr, alvo) if not os.path.isdir(p)]
    if faltando:
        logger.error(f"Modelos prontos ausentes: {faltando}")
        return []

    textos = list(X_test)
    logger.info("Rodando HateBR (ofensivo sim/não)...")
    ofensivo = _prever_binario(hatebr, textos)
    logger.info("Rodando classificador de alvo (UNTARGETED/TARGETED)...")
    dirigido = _prever_binario(alvo, textos)

    # não ofensivo -> Nenhuma; ofensivo sem alvo -> Leve; ofensivo com alvo -> Severa
    composto = [0 if o == 0 else (2 if d == 1 else 1) for o, d in zip(ofensivo, dirigido)]

    resultados = [_metricas(y_test, composto,
                            'Modelos prontos compostos (HateBR + alvo)')]

    # As duas peças, cada uma na tarefa binária para a qual foi treinada
    bin_real = [1 if r > 0 else 0 for r in y_test]
    resultados.append({
        **_metricas(bin_real, ofensivo, 'HateBR isolado (tóxico vs. não-tóxico)'),
        'tarefa': 'binária: tóxico vs. não-tóxico',
    })

    idx = [i for i, r in enumerate(y_test) if r > 0]
    if idx:
        alvo_real = [1 if y_test[i] == 2 else 0 for i in idx]
        resultados.append({
            **_metricas(alvo_real, [dirigido[i] for i in idx],
                        'Classificador de alvo isolado (Leve vs. Severa)'),
            'tarefa': 'binária, só nas tóxicas: Leve vs. Severa',
        })

    for r in resultados:
        logger.info(f"  {r['modelo'][:48]:50} F1={r['f1_macro']:.4f}")
    return resultados


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Referências sem ajuste fino')
    parser.add_argument('--modo', choices=['tudo', 'sonda', 'cloze', 'prontos'],
                        default='tudo')
    args = parser.parse_args()

    (X_train, y_train), (X_test, y_test) = get_data_for_cv_and_test(test_size=0.20)
    logger.info(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

    saida = {'timestamp': datetime.now().isoformat(),
             'n_treino': len(X_train), 'n_teste': len(X_test),
             'resultados': []}

    if args.modo in ('tudo', 'sonda'):
        logger.info("\n=== 1. SONDA LINEAR (representações congeladas) ===")
        saida['resultados'].append(rodar_sonda(X_train, y_train, X_test, y_test))

    if args.modo in ('tudo', 'cloze'):
        logger.info("\n=== 2. ZERO-SHOT POR MÁSCARA ===")
        saida['resultados'].extend(rodar_cloze(list(X_test), list(y_test)))

    if args.modo in ('tudo', 'prontos'):
        logger.info("\n=== 3. MODELOS PRONTOS DE TOXICIDADE ===")
        saida['resultados'].extend(rodar_prontos(list(X_test), list(y_test)))

    destino = os.path.join(PATHS['evaluation_dir'],
                           f"sem_finetuning_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(destino, 'w', encoding='utf-8') as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print("REFERÊNCIAS SEM AJUSTE FINO NO CORPUS — conjunto de teste retido")
    print("=" * 78)
    print(f"{'Referência':<52} {'F1 macro':>10} {'Acurácia':>10}")
    print("-" * 78)
    for r in saida['resultados']:
        print(f"{r['modelo'][:52]:<52} {r['f1_macro']:>10.4f} {r['accuracy']:>10.4f}")
    print("=" * 78)
    print(f"\nSalvo em: {destino}")


if __name__ == '__main__':
    main()
