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
    OLLAMA_HOST   (default http://127.0.0.1:11434)
    TOX_LLM_MODELS (lista separada por vírgula; sobrepõe o default)

Transparência (revisão do JBCS, M6b e m12) — só acréscimos, sem mudar prompt,
decodificação nem predição:
  - o JSON grava o prompt exato (system, exemplos few-shot, molde da mensagem),
    as opções de decodificação, as regras do parser, a resposta bruta e a
    predição de cada frase, a latência de cada requisição e a contagem de saídas
    fora do formato pedido (`saidas`, ver classificar_saida);
  - grava também a versão do Ollama e o digest do modelo (/api/version,
    /api/tags) e se ele estava na GPU (/api/ps);
  - ao fim, escreve o apêndice LaTeX com o prompt em experimento.dir_latex('apendice').
    `--so-apendice` escreve só o apêndice, sem chamar o Ollama.
  A documentação em inglês fica em docs/llm-baseline.md.
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

from app.nlp import experimento  # noqa: E402

from app.nlp.datasets.prepare_data_toxicity import get_data_for_cv_and_test
from app.nlp.config import PATHS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ======================================================================
# Configuração
# ======================================================================

RANDOM_STATE = 42
CLASS_NAMES = ['Nenhuma', 'Leve', 'Severa']

# 127.0.0.1, não "localhost": no Windows, "localhost" resolve primeiro para ::1, o
# Ollama escuta só em IPv4, e cada conexão nova esperava ~2 s pela recusa em IPv6
# antes de cair para IPv4 -- atraso do cliente que entrava na latência medida.
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')

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

# Opções de decodificação enviadas ao /api/chat. As demais (num_ctx, top_k,
# top_p, seed...) ficam no padrão do servidor Ollama. Constante única para que a
# medição de latência (evaluation/scripts/medir_latencia.py) e a documentação
# usem exatamente a requisição da linha de base.
OPCOES_OLLAMA = {"temperature": 0, "num_predict": 5}

# Categorias de saída do LLM, na ordem em que parse_label as testa. Só 'valida'
# segue o formato pedido no prompt (um único dígito); as demais são recuperadas
# pelo parser ou caem no fallback.
CATEGORIAS_SAIDA = ('valida', 'digito_extraido', 'palavra_chave', 'sem_rotulo')

REGRAS_PARSER = {
    'valida': "a resposta, sem espaços nas pontas, é exatamente '0', '1' ou '2'",
    'digito_extraido': "fora do formato, mas contém um caractere '0', '1' ou '2'; "
                       "vale o primeiro deles na ordem do texto",
    'palavra_chave': "sem dígito 0/1/2; o rótulo vem da primeira palavra-chave "
                     "encontrada, testadas nesta ordem: 'severa' -> 2, 'leve' -> 1, "
                     "'nenhuma' -> 0 (sem diferenciar maiúsculas)",
    'sem_rotulo': "vazia ou sem nada reconhecível: fallback conservador para 0 (Nenhuma)",
}


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


def classificar_saida(resposta) -> str:
    """
    Diz por qual regra parse_label chegou ao rótulo, sem alterar a predição (que
    continua sendo a de parse_label). Serve para contar as saídas fora do
    formato pedido no prompt. Categorias e regras em REGRAS_PARSER.
    """
    if not resposta:
        return 'sem_rotulo'
    if resposta.strip() in ('0', '1', '2'):
        return 'valida'
    if any(ch in '012' for ch in resposta):
        return 'digito_extraido'
    baixo = resposta.lower()
    if 'severa' in baixo or 'leve' in baixo or 'nenhuma' in baixo:
        return 'palavra_chave'
    return 'sem_rotulo'


def consultar_ollama(caminho: str, payload=None, timeout: int = 10):
    """GET (sem payload) ou POST (com payload) num endpoint do Ollama; devolve o JSON."""
    url = f"{OLLAMA_HOST}{caminho}"
    data = json.dumps(payload).encode('utf-8') if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def info_ollama(models):
    """
    Versão do servidor, digest e detalhes dos modelos avaliados (/api/tags) e o
    que está carregado agora (/api/ps: size_vram diz quanto do modelo está na
    GPU). Nunca interrompe a rodada: o que falhar fica registrado como erro.
    """
    info = {'host': OLLAMA_HOST}
    try:
        info['versao'] = consultar_ollama('/api/version').get('version')
    except Exception as e:  # noqa: BLE001
        info['erro_versao'] = str(e)
    try:
        tags = consultar_ollama('/api/tags').get('models', [])
        info['modelos'] = {
            m.get('name'): {'digest': m.get('digest'), 'size': m.get('size'),
                            'modified_at': m.get('modified_at'), 'details': m.get('details')}
            for m in tags if m.get('name') in models or m.get('model') in models
        }
    except Exception as e:  # noqa: BLE001
        info['erro_tags'] = str(e)
    try:
        ps = consultar_ollama('/api/ps').get('models', [])
        info['carregados'] = [
            {'name': m.get('name'), 'size': m.get('size'), 'size_vram': m.get('size_vram'),
             'context_length': m.get('context_length'), 'expires_at': m.get('expires_at')}
            for m in ps
        ]
    except Exception as e:  # noqa: BLE001
        info['erro_ps'] = str(e)
    return info


def descrever_prompt(fewshot):
    """O prompt exato, montado pelas mesmas funções que a avaliação usa."""
    mensagens = montar_mensagens(fewshot, '{frase}')
    return {
        'system': SYSTEM_PROMPT,
        'modelo_mensagem_usuario': mensagens[-1]['content'],
        'resposta_dos_exemplos': 'str(classe): "0", "1" ou "2"',
        'fewshot': [{'frase': txt, 'classe': int(cls)} for txt, cls in fewshot],
        'mensagens': mensagens,
        'selecao_fewshot': (
            f"{FEWSHOT_PER_CLASS} por classe, do conjunto de treino/desenvolvimento "
            f"(nunca do teste): random.Random({RANDOM_STATE}); as frases de treino são "
            "agrupadas por classe na ordem do corpus; para as classes 0, 1 e 2, nesta "
            "ordem, o grupo é embaralhado e entram as primeiras; ao fim, a ordem dos "
            "exemplos é embaralhada com o mesmo gerador (montar_fewshot)"
        ),
    }


def chamar_ollama(model: str, messages, timeout: int = 120, retries: int = 3) -> str:
    """Chama o endpoint /api/chat do Ollama (stream=False) e devolve o conteúdo."""
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": dict(OPCOES_OLLAMA),  # {"temperature": 0, "num_predict": 5}
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
    padrao = os.path.join(str(experimento.dir_resultados('toxicity_evaluation')),
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
    respostas, categorias, latencias = [], [], []
    n = len(X_test)
    t0 = time.time()
    for i, (frase, gold) in enumerate(zip(X_test, y_test), 1):
        messages = montar_mensagens(fewshot, frase)
        t_req = time.perf_counter()
        resposta = chamar_ollama(model, messages)
        latencias.append(round(time.perf_counter() - t_req, 4))
        pred = parse_label(resposta)
        # Só registro (M6b): a predição continua sendo a de parse_label.
        respostas.append(resposta)
        categorias.append(classificar_saida(resposta))
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

    # Saídas fora do formato pedido (M6b): quantas foram recuperadas pelo parser
    # e quantas caíram no fallback para Nenhuma. Não altera nenhuma predição.
    contagem = {c: categorias.count(c) for c in CATEGORIAS_SAIDA}
    metrics['saidas'] = {
        'contagem': contagem,
        'n_fora_do_formato': int(n - contagem['valida']),
        'n_fallback_nenhuma': int(contagem['sem_rotulo']),
        'regras': REGRAS_PARSER,
    }
    metrics['respostas_brutas'] = respostas
    metrics['y_pred'] = [int(p) for p in y_pred]
    # Latência de cada requisição sequencial, como a linha de base rodou (M6a).
    lat = np.asarray(latencias, dtype=float)
    metrics['latencia_requisicao_s'] = {
        'mediana': float(np.median(lat)) if len(lat) else None,
        'p95': float(np.percentile(lat, 95)) if len(lat) else None,
        'media': float(lat.mean()) if len(lat) else None,
        'amostras': latencias,
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


def salvar(resultados, models, limit, fewshot=None):
    out_dir = str(experimento.dir_resultados())
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
    # Transparência (M6b, m12): prompt exato, decodificação, parser e servidor.
    saida['experimento'] = experimento.descricao()
    if fewshot is not None:
        saida['prompt'] = descrever_prompt(fewshot)
    saida['decodificacao'] = {
        'endpoint': '/api/chat', 'stream': False, 'options': dict(OPCOES_OLLAMA),
        'nota': 'opções não listadas (num_ctx, top_k, top_p, seed...) ficam no padrão do servidor',
    }
    saida['parser'] = {
        'funcao': 'parse_label',
        'regras_em_ordem': REGRAS_PARSER,
        'tratamento_de_invalidas': 'nenhuma frase é descartada nem reenviada; '
                                   'sem rótulo reconhecível, a predição é 0 (Nenhuma)',
    }
    saida['ollama'] = info_ollama(models)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResultados salvos em: {path}")
    return path


# ----------------------------------------------------------------------
# Apêndice LaTeX com o prompt (M6b)
# ----------------------------------------------------------------------

LARGURA_APENDICE = 88  # caracteres por linha no verbatim (limite pedido: 90)


def _quebrar_linha(linha, largura=LARGURA_APENDICE, recuo='  '):
    """Quebra uma linha longa em palavras; continuações recuadas em dois espaços."""
    import textwrap
    if len(linha) <= largura:
        return [linha]
    return textwrap.wrap(linha, width=largura, subsequent_indent=recuo,
                         break_long_words=True, break_on_hyphens=False) or ['']


def _tex(texto):
    """Escapa texto comum para LaTeX (fora do verbatim)."""
    trocas = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
              '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
              '^': r'\textasciicircum{}'}
    return ''.join(trocas.get(c, c) for c in str(texto))


def _ultimo_resultado_json():
    import glob
    arquivos = glob.glob(os.path.join(str(experimento.dir_resultados()),
                                      'llm_baselines_toxicity_*.json'))
    if not arquivos:
        return None, None
    caminho = max(arquivos, key=os.path.getmtime)
    with open(caminho, encoding='utf-8') as f:
        return json.load(f), os.path.basename(caminho)


def escrever_apendice(fewshot, resultados=None, models=None, fonte=None):
    """
    Escreve apendice-llm.tex: as mensagens exatas enviadas ao modelo, em
    verbatim (compila com XeLaTeX), mais decodificação, parser e a contagem de
    saídas fora do formato quando houver resultado gravado. Sem cabeçalho de
    seção: o manuscrito decide onde e com que título entra.
    """
    mensagens = montar_mensagens(fewshot, '<sentence to classify>')
    corpo = []
    for msg in mensagens:
        corpo.append(f"[{msg['role']}]")
        for linha in msg['content'].split('\n'):
            corpo.extend(_quebrar_linha(linha))
        corpo.append('')
    while corpo and corpo[-1] == '':
        corpo.pop()
    if any('\\end{verbatim}' in l for l in corpo):
        raise ValueError('texto do prompt contém \\end{verbatim}')

    modelos = list(resultados) if resultados else list(models or [])
    opcoes = ', '.join(f'\\texttt{{{_tex(k)}}} = {v}' for k, v in OPCOES_OLLAMA.items())

    L = []
    add = L.append
    add('% Gerado por app/nlp/training/train_toxicity_llm_baselines.py')
    add('% (ao fim da rodada, ou com --so-apendice). NAO EDITAR A MAO.')
    add(f"% Rodada: {experimento.tag()}" + (f" | resultados: {fonte}" if fonte else ''))
    add(f'% Linhas longas do prompt quebradas em {LARGURA_APENDICE} caracteres; '
        'continuacoes com 2 espacos.')
    add('% O verbatim vai num figure*: no template de duas colunas do JBCS a coluna tem')
    add('% 240pt (~56 caracteres em \\footnotesize), e a largura total, 498pt (~117).')
    add('\\begin{figure*}[!t]')
    add('\\begingroup\\footnotesize')
    add('\\begin{verbatim}')
    L.extend(corpo)
    add('\\end{verbatim}')
    add('\\endgroup')
    add('\\caption{Messages sent to the generative baseline for each test sentence, in '
        'order; the role of each message is given in brackets. Lines longer than '
        f'{LARGURA_APENDICE} characters are wrapped here for typesetting, with '
        'continuation lines indented by two spaces; blank lines separate messages, and '
        'every other line break in the listing is part of the prompt. The final user '
        'message carries the sentence being classified.}')
    add('\\label{fig:prompt-llm}')
    add('\\end{figure*}')
    add('')
    add('\\begin{itemize}')
    if modelos:
        add('\\item Model: ' + ', '.join(f'\\texttt{{{_tex(m)}}}' for m in modelos)
            + ', served locally by Ollama through the \\texttt{/api/chat} endpoint, '
              'one request per sentence, without streaming.')
    else:
        add('% [modelo: sem resultado gravado -- reexecute com --models <tag>]')
    add(f'\\item Decoding: {opcoes}; all other options at the server defaults.')
    add(f'\\item Few-shot examples: {FEWSHOT_PER_CLASS} per class, drawn from the '
        'training split with a fixed seed (' + str(RANDOM_STATE) + '); the test split '
        'is never used.')
    add('\\item Parsing: if the stripped reply is exactly \\texttt{0}, \\texttt{1} or '
        '\\texttt{2}, that is the label; otherwise the first of these characters '
        'anywhere in the reply; otherwise the first keyword found, tested in the order '
        '\\emph{severa}, \\emph{leve}, \\emph{nenhuma}; otherwise the sentence is '
        'labelled None. No sentence is discarded or re-sent.')
    contagens = []
    for nome, r in (resultados or {}).items():
        s = r.get('saidas') if isinstance(r, dict) else None
        if s:
            c = s['contagem']
            contagens.append(
                f'\\texttt{{{_tex(nome)}}}: {c["valida"]} of {r["n_amostras"]} replies in '
                f'the requested format, {c["digito_extraido"]} recovered from a digit, '
                f'{c["palavra_chave"]} from a keyword, and {c["sem_rotulo"]} with no '
                'recognisable label (assigned None)')
    if contagens:
        add('\\item Replies outside the requested format: ' + '; '.join(contagens) + '.')
    else:
        add('% [contagem de saidas fora do formato: sai da rodada v1 -- depois do passo')
        add('%  llm, reexecute com --so-apendice para preencher]')
    add('\\end{itemize}')

    # Fonte .tex também em no máximo 90 colunas: fora do verbatim, quebra de
    # linha é espaço para o LaTeX; dentro dele, as linhas já vêm quebradas.
    import textwrap
    saida, no_verbatim = [], False
    for linha in L:
        if linha == '\\begin{verbatim}':
            no_verbatim = True
        elif linha == '\\end{verbatim}':
            no_verbatim = False
            saida.append(linha)
            continue
        if no_verbatim or linha.startswith('%') or len(linha) <= 90:
            saida.append(linha)
        else:
            recuo = '  ' if linha.startswith('\\item') else ''
            saida.extend(textwrap.wrap(linha, width=90, subsequent_indent=recuo,
                                       break_long_words=False, break_on_hyphens=False))

    destino = experimento.dir_latex('apendice') / 'apendice-llm.tex'
    destino.write_text('\n'.join(saida) + '\n', encoding='utf-8')
    logger.info(f"Apêndice LaTeX: {destino}")
    return destino


def main():
    parser = argparse.ArgumentParser(description="Baselines LLM para toxicidade (Ollama)")
    parser.add_argument('--models', nargs='*', default=None,
                        help="Modelos Ollama (default: llama3.1:8b gemma2:9b)")
    parser.add_argument('--limit', type=int, default=None,
                        help="Avalia só as N primeiras frases de teste (ensaio rápido)")
    parser.add_argument('--so-apendice', action='store_true',
                        help="Só escreve o apêndice LaTeX com o prompt (não chama o Ollama); "
                             "usa a contagem de saídas do resultado mais recente, se houver")
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

    if args.so_apendice:
        anterior, fonte = _ultimo_resultado_json()
        # Sem resultado gravado, só cita modelo passado explicitamente: o default
        # (llama3.1:8b e gemma2:9b) não é necessariamente o que a rodada usou.
        explicitos = args.models or (env_models.split(',') if env_models else None)
        escrever_apendice(fewshot,
                          resultados=(anterior or {}).get('results'),
                          models=(anterior or {}).get('models') or explicitos,
                          fonte=fonte)
        return

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
        caminho = salvar(resultados, models, args.limit, fewshot=fewshot)
        try:
            escrever_apendice(fewshot, resultados=resultados, models=models,
                              fonte=os.path.basename(caminho))
        except Exception as e:  # noqa: BLE001 -- o apêndice nunca derruba a rodada
            logger.warning(f"Apêndice LaTeX não escrito: {e}")
    else:
        print("Nenhum modelo avaliado com sucesso. Verifique se o 'ollama serve' está ativo "
              "e se os modelos foram baixados (ollama pull ...).")


if __name__ == "__main__":
    main()
