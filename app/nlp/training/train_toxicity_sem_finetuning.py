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
   O molde de referência é escolhido no TREINO (revisão do JBCS, m5): vence o de
   maior F1 macro sobre o conjunto de treino/desenvolvimento inteiro; o F1 nos
   folds de validação vai junto, para registro. O teste dos três moldes continua
   no JSON, por transparência, mas só o do molde escolhido no treino é a
   referência (`escolhido_no_treino`, `zero_shot_selecao`).

3. MODELOS PRONTOS DE TOXICIDADE (`--modo prontos`)
   Dois classificadores públicos em português, ambos derivados do BERTimbau,
   compostos no esquema de três classes deste trabalho:
     - `ruanchaves/bert-base-portuguese-cased-hatebr`: ofensivo (sim/não)
     - `dougtrajano/toxicity-target-classification`: UNTARGETED / TARGETED INSULT
   Composição: não ofensivo -> Nenhuma; ofensivo e sem alvo -> Leve;
   ofensivo e com alvo -> Severa. O mapeamento segue a própria definição dos
   rótulos deste corpus (Leve = ofensividade indireta; Severa = dirigida).
   Mede o que se obteria sem construir corpus algum.
   Os pesos são lidos de cópias locais em data/models/ (baixadas com
   `hf download --local-dir`); a revisão (commit do Hub) de cada cópia é lida do
   .cache/huggingface/download/ que o download deixa e gravada no JSON
   (`modelos_externos`). Ver docs/external-models.md.

Todas as referências são avaliadas no MESMO conjunto de teste retido do ajuste
fino. A sonda linear, por ser treinada nos nossos folds, também reporta
validação cruzada; as demais não têm treino sobre este corpus e por isso não
têm coluna de CV.

Uso:
    python -m app.nlp.training.train_toxicity_sem_finetuning              # tudo
    python -m app.nlp.training.train_toxicity_sem_finetuning --modo sonda
    # ensaio fora da rodada (o JSON não entra em results/<tag>/):
    python -m app.nlp.training.train_toxicity_sem_finetuning --modo cloze --saida <pasta>
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp import experimento  # noqa: E402

from app.nlp.datasets.prepare_data_toxicity import get_data_for_cv_and_test
from app.nlp.utils.data_utils import get_kfold_split
from app.nlp.config import MODEL_CONFIG, PATHS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLASSES = ['Nenhuma', 'Leve', 'Severa']
K_FOLDS = 5
RANDOM_STATE = 42
MODELOS_PRONTOS_DIR = os.path.join(project_root, 'data', 'models')

# Modelos prontos: identificador no Hugging Face, pasta local e a revisão
# (commit) da cópia usada nos resultados, conferida contra o Hub em 2026-11-19
# (docs/external-models.md). Se a cópia local for de outra revisão, o script
# avisa e grava a revisão encontrada -- não troca de modelo em silêncio.
MODELOS_PRONTOS = {
    'hatebr': {
        'hf_id': 'ruanchaves/bert-base-portuguese-cased-hatebr',
        'pasta': 'ruanchaves_bert_base_hatebr',
        'revisao_esperada': 'ed31720b2f3c0b6cefe81cf267751b255b377a7d',
        'rotulos': {0: 'não ofensivo', 1: 'ofensivo'},
    },
    'alvo': {
        'hf_id': 'dougtrajano/toxicity-target-classification',
        'pasta': 'dougtrajano_toxicity_target_classification',
        'revisao_esperada': '885d2698f9b68f0fc90850fe7fc4dc7cc970f2f5',
        'rotulos': {0: 'UNTARGETED', 1: 'TARGETED INSULT'},
    },
}

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


def rodar_sonda(X_train, y_train, X_test, y_test, grupos=None):
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
                        random_state=RANDOM_STATE, groups=grupos), 1
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

def _prever_cloze(modelo, tokenizador, ancoras_ids, molde, textos, batch_size=32):
    """Classe predita para cada texto: a âncora de maior probabilidade na máscara."""
    preds = []
    with torch.no_grad():
        for i in range(0, len(textos), batch_size):
            lote = [molde.format(frase=t) for t in textos[i:i + batch_size]]
            entradas = tokenizador(lote, padding=True, truncation=True,
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
    return preds


def _f1_treino_cloze(X_train, y_train, preds_treino, grupos):
    """
    F1 do molde no treino inteiro (critério de escolha) e nos folds de validação
    da rodada (só registro: sem ajuste, cada fold é apenas um subconjunto).
    """
    y = np.asarray(y_train)
    p = np.asarray(preds_treino)
    saida = {
        'n': int(len(y)),
        'f1_macro': float(f1_score(y, p, average='macro', zero_division=0)),
        'accuracy': float(accuracy_score(y, p)),
    }
    try:
        f1s = [float(f1_score(y[va], p[va], average='macro', zero_division=0))
               for _, va in get_kfold_split(np.array(X_train), y, n_splits=K_FOLDS,
                                            random_state=RANDOM_STATE, groups=grupos)]
        saida.update(cv_f1_por_fold=f1s, cv_f1_mean=float(np.mean(f1s)),
                     cv_f1_std=float(np.std(f1s)))
    except Exception as e:  # noqa: BLE001 -- os folds são só registro
        logger.warning(f"F1 por fold do zero-shot não calculado: {e}")
        saida['cv_erro'] = str(e)
    return saida


def resumo_selecao_cloze(resultados):
    """Resumo da escolha do molde no treino, para o topo do JSON (m5)."""
    com_treino = [r for r in resultados if 'treino' in r]
    if not com_treino:
        return None
    escolhido = next((r for r in com_treino if r.get('escolhido_no_treino')), None)
    if escolhido is None:
        return None
    return {
        'molde_escolhido_no_treino': escolhido['molde'],
        'criterio': ('maior F1 macro no conjunto de treino/desenvolvimento inteiro; '
                     'empate: o primeiro na ordem de MOLDES_CLOZE. O teste não entra '
                     'na escolha.'),
        'n_treino': escolhido['treino']['n'],
        'f1_treino_por_molde': {r['molde']: r['treino']['f1_macro'] for r in com_treino},
        'cv_f1_treino_por_molde': {
            r['molde']: {'media': r['treino'].get('cv_f1_mean'),
                         'dp': r['treino'].get('cv_f1_std')} for r in com_treino},
        'f1_teste_do_escolhido': escolhido['f1_macro'],
        'f1_teste_por_molde_so_registro': {r['molde']: r['f1_macro'] for r in com_treino},
        'nota': ('Só o teste do molde escolhido no treino é a referência; o teste dos '
                 'demais fica no JSON por transparência e não entra em tabela nem figura.'),
    }


def rodar_cloze(X_test, y_test, X_train=None, y_train=None, grupos=None):
    """
    Zero-shot por máscara com os moldes de MOLDES_CLOZE.

    Com X_train/y_train (o que main faz), cada molde também é avaliado no treino
    e o de maior F1 macro no treino vira a referência (`escolhido_no_treino`).
    Sem eles, comportamento antigo: só o teste, sem escolha.
    """
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
        preds = _prever_cloze(modelo, tokenizador, ancoras_ids, molde, list(X_test))
        r = _metricas(y_test, preds, f'Zero-shot máscara: "{molde}"')
        r['molde'] = molde
        if X_train is not None and y_train is not None:
            preds_tr = _prever_cloze(modelo, tokenizador, ancoras_ids, molde, list(X_train))
            r['treino'] = _f1_treino_cloze(X_train, y_train, preds_tr, grupos)
            logger.info(f"  {molde[:40]:42} F1 treino={r['treino']['f1_macro']:.4f} | "
                        f"teste={r['f1_macro']:.4f}")
        else:
            logger.info(f"  {molde[:40]:42} F1={r['f1_macro']:.4f}")
        resultados.append(r)

    # m5: a referência é o molde de maior F1 no TREINO (max devolve o primeiro
    # em caso de empate, isto é, a ordem de MOLDES_CLOZE). O teste dos outros
    # moldes fica marcado como só registro.
    if resultados and all('treino' in r for r in resultados):
        i_esc = max(range(len(resultados)), key=lambda i: resultados[i]['treino']['f1_macro'])
        for i, r in enumerate(resultados):
            r['escolhido_no_treino'] = (i == i_esc)
            r['papel'] = ('referência: molde escolhido no treino' if i == i_esc else
                          'só registro: molde não escolhido no treino (teste não usado '
                          'em nenhuma escolha)')
        logger.info(f"  Molde escolhido no treino: {resultados[i_esc]['molde']} "
                    f"(teste: F1={resultados[i_esc]['f1_macro']:.4f})")

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


def revisao_local(pasta):
    """
    Revisão (commit do Hub) de uma cópia baixada com `hf download --local-dir`.

    O download grava .cache/huggingface/download/<arquivo>.metadata com três
    linhas: commit, etag e data. Para arquivos LFS (os pesos), o etag é o sha256
    do arquivo. Devolve {arquivo: {'commit', 'etag'}} para os arquivos que
    definem o modelo.
    """
    meta = os.path.join(pasta, '.cache', 'huggingface', 'download')
    saida = {}
    for nome in ('config.json', 'pytorch_model.bin', 'model.safetensors',
                 'tokenizer.json', 'vocab.txt'):
        arq = os.path.join(meta, nome + '.metadata')
        if os.path.isfile(arq):
            with open(arq, encoding='utf-8') as f:
                linhas = f.read().split()
            if len(linhas) >= 2:
                saida[nome] = {'commit': linhas[0], 'etag': linhas[1]}
    return saida


def info_modelos_externos():
    """Identificador, revisão usada e conferência contra a revisão esperada (M6c, m12)."""
    infos = []
    for papel, m in MODELOS_PRONTOS.items():
        pasta = os.path.join(MODELOS_PRONTOS_DIR, m['pasta'])
        arquivos = revisao_local(pasta)
        commits = sorted({v['commit'] for v in arquivos.values()})
        usada = commits[0] if len(commits) == 1 else None
        info = {
            'papel': papel,
            'hf_id': m['hf_id'],
            'url': f"https://huggingface.co/{m['hf_id']}",
            'pasta_local': os.path.relpath(pasta, project_root).replace(os.sep, '/'),
            'revisao_usada': usada,
            'revisoes_encontradas': commits,
            'revisao_esperada': m['revisao_esperada'],
            'confere': usada == m['revisao_esperada'],
            'pesos_sha256': (arquivos.get('pytorch_model.bin')
                             or arquivos.get('model.safetensors') or {}).get('etag'),
            'rotulos': m['rotulos'],
        }
        if not info['confere']:
            logger.warning(f"Revisão local de {m['hf_id']} = {commits or 'desconhecida'}; "
                           f"esperada {m['revisao_esperada']} (docs/external-models.md)")
        infos.append(info)
    return infos


def info_modelo_base():
    """Revisão do BERTimbau base no cache do Hugging Face usado pelo projeto."""
    nome = MODEL_CONFIG['base_model']
    cache = MODEL_CONFIG.get('cache_dir') or ''
    ref = os.path.join(cache, 'models--' + nome.replace('/', '--'), 'refs', 'main')
    revisao = None
    if os.path.isfile(ref):
        with open(ref, encoding='utf-8') as f:
            revisao = f.read().strip() or None
    return {'hf_id': nome, 'url': f'https://huggingface.co/{nome}',
            'revisao_no_cache': revisao}


def rodar_prontos(X_test, y_test):
    hatebr = os.path.join(MODELOS_PRONTOS_DIR, MODELOS_PRONTOS['hatebr']['pasta'])
    alvo = os.path.join(MODELOS_PRONTOS_DIR, MODELOS_PRONTOS['alvo']['pasta'])

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
    parser.add_argument('--saida', default=None,
                        help='Pasta do JSON (padrão: experimento.dir_resultados()); para '
                             'ensaios que não devem entrar nos resultados da rodada')
    args = parser.parse_args()

    (X_train, y_train, g_train), (X_test, y_test) = get_data_for_cv_and_test(
        test_size=0.20, return_groups=True)
    logger.info(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

    saida = {'timestamp': datetime.now().isoformat(),
             'n_treino': len(X_train), 'n_teste': len(X_test),
             'resultados': []}

    if args.modo in ('tudo', 'sonda'):
        logger.info("\n=== 1. SONDA LINEAR (representações congeladas) ===")
        saida['resultados'].append(rodar_sonda(X_train, y_train, X_test, y_test, grupos=g_train))

    if args.modo in ('tudo', 'sonda', 'cloze'):
        saida['modelo_base'] = info_modelo_base()

    if args.modo in ('tudo', 'cloze'):
        logger.info("\n=== 2. ZERO-SHOT POR MÁSCARA ===")
        # m5: o molde é escolhido no treino; o teste dos três fica só para registro.
        res_cloze = rodar_cloze(list(X_test), list(y_test),
                                X_train=list(X_train), y_train=list(y_train), grupos=g_train)
        saida['resultados'].extend(res_cloze)
        saida['zero_shot_selecao'] = resumo_selecao_cloze(res_cloze)

    if args.modo in ('tudo', 'prontos'):
        logger.info("\n=== 3. MODELOS PRONTOS DE TOXICIDADE ===")
        try:
            saida['modelos_externos'] = info_modelos_externos()
        except Exception as e:  # noqa: BLE001 -- registro nunca derruba a etapa
            logger.warning(f"Revisões dos modelos prontos não registradas: {e}")
        saida['resultados'].extend(rodar_prontos(list(X_test), list(y_test)))

    pasta_saida = Path(args.saida) if args.saida else experimento.dir_resultados()
    pasta_saida.mkdir(parents=True, exist_ok=True)
    destino = str(pasta_saida / f"sem_finetuning_{datetime.now():%Y%m%d_%H%M%S}.json")
    saida['experimento'] = experimento.descricao()
    with open(destino, 'w', encoding='utf-8') as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print("REFERÊNCIAS SEM AJUSTE FINO NO CORPUS — conjunto de teste retido")
    print("=" * 78)
    print(f"{'Referência':<52} {'F1 macro':>10} {'Acurácia':>10}")
    print("-" * 78)
    for r in saida['resultados']:
        marca = ''
        if 'escolhido_no_treino' in r:
            marca = ' *' if r['escolhido_no_treino'] else ' (registro)'
        print(f"{(r['modelo'][:46] + marca):<52} {r['f1_macro']:>10.4f} {r['accuracy']:>10.4f}")
    print("=" * 78)
    if saida.get('zero_shot_selecao'):
        print("* zero-shot: molde escolhido no treino; '(registro)' = teste só para "
              "transparência")
    print(f"\nSalvo em: {destino}")


if __name__ == '__main__':
    main()
