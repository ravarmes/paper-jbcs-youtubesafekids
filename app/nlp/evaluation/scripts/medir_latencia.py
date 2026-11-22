"""
Latência de inferência por página de resultados: BERTimbau ajustado e LLM
generativo medidos na MESMA máquina (revisão do JBCS: M6a e M5d).

O QUE MEDE
----------
Uma página de resultados tem 10 vídeos, e de cada vídeo o serviço classifica k
frases. Para k = 3 (30 frases) e k = 12 (120 frases), com frases REAIS do teste
retido da rodada (sorteio fixo, semente 42; as 30 frases de k = 3 são as 30
primeiras das 120, então os dois modelos e os dois k usam as mesmas frases):

  (a) BERTimbau ajustado -- o modelo de referência da rodada
      (experimento.caminho_modelo_referencia) ou --modelo. A página inteira vai
      num lote só, pela API do projeto (BertimbauToxicity.predict_batch:
      tokenização com padding até max_length = 128, forward, softmax, rótulos).
      Em GPU e em CPU, com aquecimento descartado e N repetições; tempo de
      parede com time.perf_counter (e torch.cuda.synchronize na GPU). A carga do
      modelo é medida à parte e não entra no tempo da página.

  (b) O LLM via Ollama, com o MESMO prompt few-shot da linha de base
      (train_toxicity_llm_baselines: SYSTEM_PROMPT, montar_fewshot,
      montar_mensagens) e as mesmas frases, em modos `formato:concorrência`:
        livre:1     sequencial, uma requisição por frase -- exatamente a
                    requisição da linha de base (OPCOES_OLLAMA: temperature 0,
                    num_predict 5)
        restrita:1  sequencial, saída restrita por esquema JSON (`format`) com as
                    três classes e num_predict pequeno (--num-predict-restrito)
        restrita:c  c requisições concorrentes (threads), saída restrita
      Grava o modelo e o digest, a versão do Ollama, o num_ctx (o que o
      servidor informa) e se o modelo estava na GPU (/api/ps: size_vram contra
      size), as estatísticas que o próprio Ollama devolve por requisição e a
      contagem de saídas fora do formato. O prompt não muda em modo nenhum: só a
      decodificação (restrita) e o paralelismo.

ORDEM E ISOLAMENTO
------------------
O LLM é medido PRIMEIRO, antes de qualquer uso de CUDA neste processo, para que
o contexto CUDA do PyTorch não tire VRAM do Ollama. Ao fim, o modelo é
descarregado do Ollama (keep_alive = 0) para que o BERTimbau na GPU não dispute
memória com ele; --manter-llm desliga a descarga. Cada modelo é medido sozinho
na máquina, que é a comparação justa: o serviço rodaria um ou outro.

O QUE NÃO MEDE
--------------
  - A recuperação da transcrição: componente externo ao classificador. Fica no
    JSON como "não medido", sem valor estimado.
  - O orçamento de latência de uma busca: fica nulo até haver referência com
    fonte (plano P5).

SAÍDAS
------
  latencia_<timestamp>.json em experimento.dir_resultados() (ou --saida)
  tab-latencia.tex           em experimento.dir_latex('tabelas') (ou --saida)

Uso (de src/, com a GPU livre e o Ollama ativo e ocioso):
    python -m app.nlp.evaluation.scripts.medir_latencia
    python -m app.nlp.evaluation.scripts.medir_latencia --sem-llm
    python -m app.nlp.evaluation.scripts.medir_latencia --sem-gpu --sem-llm --repeticoes 3
    python -m app.nlp.evaluation.scripts.medir_latencia --so-tabela <latencia_X.json>

Com --sem-gpu, exporte também CUDA_VISIBLE_DEVICES=-1 no shell: o pacote
app.nlp.evaluation importa o torch antes deste módulo rodar (a variável ainda
vale, porque o CUDA só inicializa no primeiro uso, mas no shell não há dúvida).
"""

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

VIDEOS_POR_PAGINA = 10
KS_PADRAO = [3, 12]
SEMENTE = 42
MODOS_LLM_PADRAO = ['livre:1', 'restrita:1', 'restrita:4', 'restrita:8']
MODELO_OLLAMA_PADRAO = 'llama3.1:8b'
NUM_PREDICT_RESTRITO = 16   # '{"classe": 2}' cabe com folga; truncado = fora do esquema

# Saída restrita: o Ollama compila o esquema em gramática e só deixa o modelo
# gerar um objeto com a chave "classe" valendo 0, 1 ou 2.
ESQUEMA_RESTRITO = {
    'type': 'object',
    'properties': {'classe': {'type': 'integer', 'enum': [0, 1, 2]}},
    'required': ['classe'],
}

RECUPERACAO_TRANSCRICAO = {
    'medido': False,
    'status': 'não medido',
    'valor_s': None,
    'motivo': ('componente externo ao classificador (provedor de transcrição); não '
               'faz parte do que este script executa e não tem valor estimado aqui'),
}

ORCAMENTO_LATENCIA = {
    'valor_s': None,
    'status': 'não definido: precisa de referência com fonte (plano P5); não inventar',
}


# ---------------------------------------------------------------------------
# Estatística e utilidades
# ---------------------------------------------------------------------------

def _percentil(valores, q):
    """Percentil com interpolação linear (mesma regra do numpy.percentile)."""
    v = sorted(float(x) for x in valores)
    if not v:
        return None
    pos = (len(v) - 1) * q / 100.0
    baixo = int(pos)
    alto = min(baixo + 1, len(v) - 1)
    return v[baixo] + (v[alto] - v[baixo]) * (pos - baixo)


def _mediana(valores):
    return _percentil(valores, 50)


def resumir(tempos, n_frases):
    """Resumo dos tempos de parede de uma página (segundos)."""
    med = _mediana(tempos)
    return {
        'n_frases': int(n_frases),
        'repeticoes': len(tempos),
        'mediana_s': med,
        'p95_s': _percentil(tempos, 95),
        'media_s': (sum(tempos) / len(tempos)) if tempos else None,
        'min_s': min(tempos) if tempos else None,
        'max_s': max(tempos) if tempos else None,
        'por_frase_mediana_ms': (med / n_frases * 1000.0) if (med is not None and n_frases) else None,
        'amostras_s': [round(float(t), 6) for t in tempos],
    }


def _rodar(cmd, timeout=20):
    """Roda um comando de consulta e devolve a saída, ou None se falhar."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    except Exception:  # noqa: BLE001
        return None


def _caminho_publico(caminho):
    """Caminho relativo a src/ quando possível: o JSON é público e não leva home de ninguém."""
    try:
        return Path(caminho).resolve().relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:
        return Path(caminho).name


# ---------------------------------------------------------------------------
# Hardware e versões
# ---------------------------------------------------------------------------

def nome_cpu():
    if sys.platform.startswith('win'):
        for cmd in (['powershell', '-NoProfile', '-Command',
                     '(Get-CimInstance Win32_Processor).Name'],
                    ['wmic', 'cpu', 'get', 'name']):
            s = _rodar(cmd)
            if s:
                linhas = [l.strip() for l in s.splitlines()
                          if l.strip() and l.strip().lower() != 'name']
                if linhas:
                    return linhas[0]
    elif os.path.exists('/proc/cpuinfo'):
        with open('/proc/cpuinfo', encoding='utf-8', errors='ignore') as f:
            for l in f:
                if l.startswith('model name'):
                    return l.split(':', 1)[1].strip()
    elif sys.platform == 'darwin':
        s = _rodar(['sysctl', '-n', 'machdep.cpu.brand_string'])
        if s:
            return s
    return platform.processor() or None


def memoria_ram_gb():
    if sys.platform.startswith('win'):
        s = _rodar(['powershell', '-NoProfile', '-Command',
                    '(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory'])
        try:
            return round(int(s) / 2 ** 30, 1)
        except (TypeError, ValueError):
            return None
    if os.path.exists('/proc/meminfo'):
        with open('/proc/meminfo', encoding='utf-8') as f:
            for l in f:
                if l.startswith('MemTotal'):
                    return round(int(l.split()[1]) / 2 ** 20, 1)
    return None


def gpu_nvidia_smi():
    """Nome, memória e driver pelo nvidia-smi (não inicializa CUDA neste processo)."""
    s = _rodar(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                '--format=csv,noheader'])
    if not s:
        return None
    partes = [p.strip() for p in s.splitlines()[0].split(',')]
    return {'nome': partes[0] if partes else None,
            'memoria_total': partes[1] if len(partes) > 1 else None,
            'driver': partes[2] if len(partes) > 2 else None}


def info_hardware():
    return {
        'cpu': nome_cpu(),
        'cpu_nucleos_logicos': os.cpu_count(),
        'memoria_ram_gb': memoria_ram_gb(),
        'gpu': gpu_nvidia_smi(),
        'sistema': platform.platform(),
    }


def versoes_bibliotecas():
    import importlib
    v = {'python': platform.python_version()}
    for nome in ('torch', 'transformers', 'tokenizers', 'numpy'):
        try:
            v[nome] = importlib.import_module(nome).__version__
        except Exception:  # noqa: BLE001
            v[nome] = None
    try:
        import torch
        v['cuda_do_torch'] = torch.version.cuda
    except Exception:  # noqa: BLE001
        pass
    return v


# ---------------------------------------------------------------------------
# Frases das páginas
# ---------------------------------------------------------------------------

def montar_paginas(ks, videos, semente):
    """
    Sorteia videos x max(k) frases do teste retido (sem reposição quando cabe);
    a página de k usa as primeiras videos x k. Devolve também o treino, de onde
    sai o few-shot do LLM.
    """
    from app.nlp.datasets.prepare_data_toxicity import get_data_for_cv_and_test

    (X_train, y_train), (X_test, _y_test) = get_data_for_cv_and_test(test_size=0.20)
    n_max = videos * max(ks)
    rng = random.Random(semente)
    if n_max <= len(X_test):
        idx = rng.sample(range(len(X_test)), n_max)
        reposicao = False
    else:
        idx = [rng.randrange(len(X_test)) for _ in range(n_max)]
        reposicao = True
    paginas = {k: [X_test[i] for i in idx[:videos * k]] for k in ks}
    frases_info = {
        'origem': 'teste retido da rodada (get_data_for_cv_and_test)',
        'n_teste': len(X_test),
        'semente': semente,
        'com_reposicao': reposicao,
        'indices_no_teste': idx,
        'caracteres_por_frase_media': {
            str(k): sum(len(f) for f in fs) / len(fs) for k, fs in paginas.items()},
    }
    return paginas, frases_info, (X_train, y_train)


# ---------------------------------------------------------------------------
# (a) BERTimbau
# ---------------------------------------------------------------------------

def medir_bertimbau(caminho, dispositivo, paginas, repeticoes, aquecimento, lote, threads):
    import gc
    import torch
    from app.nlp.models.bertimbau_toxicity import BertimbauToxicity

    if threads:
        torch.set_num_threads(threads)
    cuda = dispositivo == 'cuda'
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    modelo = BertimbauToxicity(model_path=str(caminho), device=dispositivo)
    modelo.model.eval()
    if cuda:
        torch.cuda.synchronize()
    carga = time.perf_counter() - t0

    out = {
        'dispositivo': dispositivo,
        'carga_modelo_s': carga,
        'threads_cpu_torch': torch.get_num_threads(),
        'api': 'BertimbauToxicity.predict_batch (padding max_length, softmax, rótulos)',
        'max_length': modelo.max_length,
        'paginas': {},
    }
    if cuda:
        out['gpu_torch'] = torch.cuda.get_device_name(0)
        out['gpu_memoria_total_gb'] = round(
            torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 2)

    for k, frases in paginas.items():
        bs = lote or len(frases)
        for _ in range(aquecimento):
            modelo.predict_batch(frases, batch_size=bs)
        tempos = []
        for _ in range(repeticoes):
            if cuda:
                torch.cuda.synchronize()
            t = time.perf_counter()
            modelo.predict_batch(frases, batch_size=bs)
            if cuda:
                torch.cuda.synchronize()
            tempos.append(time.perf_counter() - t)
        tokens = [len(modelo.tokenizer.tokenize(f)) for f in frases]
        out['paginas'][str(k)] = dict(
            resumir(tempos, len(frases)), lote=bs,
            tokens_por_frase={'media': sum(tokens) / len(tokens), 'max': max(tokens),
                              'nota': f'cada frase é preenchida até {modelo.max_length}'})
        print(f"  BERTimbau {dispositivo} k={k}: mediana "
              f"{out['paginas'][str(k)]['mediana_s']:.4f} s/página", flush=True)

    if cuda:
        out['memoria_gpu_pico_mb'] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    del modelo
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# (b) LLM via Ollama
# ---------------------------------------------------------------------------

def _interpretar_restrita(conteudo, base):
    """Saída restrita: válida se for JSON com 'classe' em {0,1,2}; senão, parser da linha de base."""
    try:
        obj = json.loads(conteudo)
        v = obj.get('classe') if isinstance(obj, dict) else None
        if isinstance(v, int) and not isinstance(v, bool) and v in (0, 1, 2):
            return v, 'valida'
    except (ValueError, TypeError):
        pass
    return base.parse_label(conteudo), 'fora_do_esquema'


def _entrada_ps(base, modelo):
    try:
        for m in base.consultar_ollama('/api/ps').get('models', []):
            if modelo in (m.get('name'), m.get('model')):
                return m
    except Exception:  # noqa: BLE001
        return None
    return None


def _contexto_do_show(show):
    info = (show or {}).get('model_info') or {}
    for chave, valor in info.items():
        if chave.endswith('.context_length'):
            return valor
    return None


def _num_ctx_do_modelfile(show):
    for linha in ((show or {}).get('parameters') or '').splitlines():
        partes = linha.split()
        if len(partes) >= 2 and partes[0] == 'num_ctx':
            return partes[1]
    return None


def medir_llm(modelo_ollama, paginas, fewshot, modos, repeticoes, aquecimento,
              num_predict_restrito, timeout, manter_carregado):
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor
    from app.nlp.training import train_toxicity_llm_baselines as base

    saida = {'modelo': modelo_ollama, 'host': base.OLLAMA_HOST}
    try:
        saida['versao_ollama'] = base.consultar_ollama('/api/version').get('version')
    except Exception as e:  # noqa: BLE001
        saida['erro'] = f'Ollama inacessível em {base.OLLAMA_HOST}: {e}'
        print(f"  {saida['erro']} -- parte do LLM pulada", flush=True)
        return saida

    try:
        saida['carregados_no_inicio'] = [m.get('name') for m in
                                         base.consultar_ollama('/api/ps').get('models', [])]
    except Exception:  # noqa: BLE001
        pass

    def payload(frase, restrita):
        p = {'model': modelo_ollama, 'messages': base.montar_mensagens(fewshot, frase),
             'stream': False, 'options': dict(base.OPCOES_OLLAMA)}
        if restrita:
            p['format'] = ESQUEMA_RESTRITO
            p['options']['num_predict'] = num_predict_restrito
        return p

    def requisitar(frase, restrita):
        t = time.perf_counter()
        try:
            corpo = base.consultar_ollama('/api/chat', payload(frase, restrita), timeout=timeout)
        except Exception as e:  # noqa: BLE001 -- conta como erro, não derruba a medição
            return {'s': time.perf_counter() - t, 'erro': str(e)}
        dt = time.perf_counter() - t
        conteudo = corpo.get('message', {}).get('content', '')
        if restrita:
            pred, categoria = _interpretar_restrita(conteudo, base)
        else:
            pred, categoria = base.parse_label(conteudo), base.classificar_saida(conteudo)
        stats = {k: corpo.get(k) for k in ('total_duration', 'load_duration',
                                            'prompt_eval_count', 'prompt_eval_duration',
                                            'eval_count', 'eval_duration')}
        return {'s': dt, 'conteudo': conteudo, 'pred': pred, 'categoria': categoria,
                'stats': stats}

    # Aquecimento geral: carrega o modelo; a carga sai da primeira resposta.
    primeira = paginas[min(paginas)]
    t0 = time.perf_counter()
    r0 = requisitar(primeira[0], False)
    saida['primeira_requisicao_s'] = time.perf_counter() - t0
    if 'erro' in r0:
        saida['erro'] = f"primeira requisição falhou: {r0['erro']}"
        return saida
    ld = (r0.get('stats') or {}).get('load_duration')
    saida['carga_modelo_s'] = ld / 1e9 if ld else None

    ps = _entrada_ps(base, modelo_ollama) or {}
    try:
        show = base.consultar_ollama('/api/show', {'model': modelo_ollama})
    except Exception:  # noqa: BLE001
        show = {}
    try:
        tags = base.consultar_ollama('/api/tags').get('models', [])
        digest = next((m.get('digest') for m in tags
                       if modelo_ollama in (m.get('name'), m.get('model'))), None)
    except Exception:  # noqa: BLE001
        digest = None
    tam, vram = ps.get('size'), ps.get('size_vram')
    if tam and vram is not None:
        na_gpu = 'total' if vram >= tam else ('parcial' if vram > 0 else 'nao')
    else:
        na_gpu = None
    saida.update({
        'digest': digest,
        'detalhes': (show or {}).get('details'),
        'memoria': {'size_bytes': tam, 'size_vram_bytes': vram, 'na_gpu': na_gpu,
                    'fracao_na_gpu': (vram / tam) if (tam and vram is not None) else None,
                    'fonte': '/api/ps depois do aquecimento'},
        'num_ctx': {
            'enviado_nas_opcoes': None,
            'contexto_em_uso_api_ps': ps.get('context_length'),
            'num_ctx_no_modelfile': _num_ctx_do_modelfile(show),
            'contexto_de_treino_do_modelo': _contexto_do_show(show),
            'OLLAMA_CONTEXT_LENGTH_neste_processo': os.environ.get('OLLAMA_CONTEXT_LENGTH'),
            'OLLAMA_NUM_PARALLEL_neste_processo': os.environ.get('OLLAMA_NUM_PARALLEL'),
            'nota': ('a linha de base não envia num_ctx: vale o padrão do servidor. '
                     '/api/ps, quando o servidor informa context_length, é a fonte direta; '
                     'as variáveis de ambiente listadas são as deste processo, não '
                     'necessariamente as do servidor'),
        },
        'requisicao': {
            'endpoint': '/api/chat', 'stream': False,
            'opcoes_livre': dict(base.OPCOES_OLLAMA),
            'opcoes_restrita': dict(base.OPCOES_OLLAMA, num_predict=num_predict_restrito),
            'format_restrita': ESQUEMA_RESTRITO,
            'prompt': 'idêntico ao da linha de base (SYSTEM_PROMPT + few-shot + "Frase: ...")',
        },
        'modos': {},
    })

    for modo in modos:
        formato, conc = modo.split(':')
        restrita, conc = (formato == 'restrita'), int(conc)
        for f in primeira[:aquecimento]:
            requisitar(f, restrita)
        res = {'formato': formato, 'concorrencia': conc, 'paginas': {}}
        for k, frases in paginas.items():
            tempos, todas, ultima = [], [], []
            for _ in range(repeticoes):
                t = time.perf_counter()
                if conc == 1:
                    saidas_rep = [requisitar(f, restrita) for f in frases]
                else:
                    with ThreadPoolExecutor(max_workers=conc) as ex:
                        saidas_rep = list(ex.map(lambda f: requisitar(f, restrita), frases))
                tempos.append(time.perf_counter() - t)
                todas.extend(saidas_rep)
                ultima = saidas_rep
            ok = [r for r in todas if 'erro' not in r]
            lat = [r['s'] for r in ok]

            def med_stat(chave, escala=1.0):
                vals = [r['stats'][chave] for r in ok if r['stats'].get(chave) is not None]
                m = _mediana(vals)
                return m / escala if m is not None else None

            res['paginas'][str(k)] = dict(
                resumir(tempos, len(frases)),
                requisicao_mediana_s=_mediana(lat), requisicao_p95_s=_percentil(lat, 95),
                n_requisicoes=len(todas), n_erros=len(todas) - len(ok),
                erros_exemplo=[r['erro'] for r in todas if 'erro' in r][:3],
                saidas=dict(Counter(r['categoria'] for r in ok)),
                ollama_mediana={
                    'prompt_eval_count': med_stat('prompt_eval_count'),
                    'prompt_eval_ms': med_stat('prompt_eval_duration', 1e6),
                    'eval_count': med_stat('eval_count'),
                    'eval_ms': med_stat('eval_duration', 1e6),
                    'total_ms': med_stat('total_duration', 1e6),
                },
                predicoes_ultima_repeticao=[r.get('pred') for r in ultima],
            )
            print(f"  {modelo_ollama} {modo} k={k}: mediana "
                  f"{res['paginas'][str(k)]['mediana_s']:.2f} s/página", flush=True)
        saida['modos'][modo] = res

    # Diagnóstico (não é métrica do artigo): a saída restrita e a concorrência
    # mudam a classe predita em relação à requisição da linha de base?
    ref = saida['modos'].get('livre:1')
    if ref:
        for modo, res in saida['modos'].items():
            if modo == 'livre:1':
                continue
            res['concordancia_com_livre_1'] = {}
            for k, pag in res['paginas'].items():
                a = ref['paginas'][k]['predicoes_ultima_repeticao']
                b = pag['predicoes_ultima_repeticao']
                pares = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
                res['concordancia_com_livre_1'][k] = (
                    sum(x == y for x, y in pares) / len(pares) if pares else None)

    if not manter_carregado:
        try:
            base.consultar_ollama('/api/generate', {'model': modelo_ollama, 'keep_alive': 0,
                                                    'stream': False}, timeout=60)
            saida['descarregado_ao_fim'] = True
        except Exception as e:  # noqa: BLE001
            saida['descarregado_ao_fim'] = f'falhou: {e}'
    return saida


# ---------------------------------------------------------------------------
# Tabela LaTeX
# ---------------------------------------------------------------------------

def _tex(texto):
    trocas = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
              '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
              '^': r'\textasciicircum{}'}
    return ''.join(trocas.get(c, c) for c in str(texto))


def _seg(v):
    if v is None:
        return '--'
    if v < 0.1:
        return f'{v:.3f}'
    if v < 10:
        return f'{v:.2f}'
    return f'{v:.1f}'


def _ms(v):
    if v is None:
        return '--'
    return f'{v:.1f}' if v < 10 else f'{v:.0f}'


def linhas_tabela(d):
    """(modelo, configuração, páginas) para cada linha medida, na ordem da tabela."""
    linhas = []
    bert = d.get('bertimbau') or {}
    for disp, rotulo in (('gpu', 'GPU, one batch'), ('cpu', 'CPU, one batch')):
        r = bert.get(disp) or {}
        if r.get('paginas'):
            linhas.append(('BERTimbau (fine-tuned)', rotulo, r['paginas']))
    llm = d.get('llm') or {}
    for modo, r in (llm.get('modos') or {}).items():
        conc = r['concorrencia']
        rotulo = ('sequential' if conc == 1 else f'{conc} concurrent')
        rotulo += ', free' if r['formato'] == 'livre' else ', constrained'
        linhas.append((_tex(llm.get('modelo', 'LLM')), rotulo, r['paginas']))
    return linhas


def escrever_tabela(d, destino, fonte=None):
    """Escreve tab-latencia.tex a partir do JSON; devolve o caminho ou None."""
    linhas = linhas_tabela(d)
    if not linhas:
        print('  Nenhuma medição para tabelar; tab-latencia.tex não escrito.')
        return None
    cfg = d['configuracao']
    ks = [str(k) for k in cfg['ks']]
    videos = cfg['videos_por_pagina']
    k_max = ks[-1]

    hw = d.get('hardware') or {}
    gpu = (hw.get('gpu') or {}).get('nome')
    bert_gpu = (d.get('bertimbau') or {}).get('gpu') or {}
    if bert_gpu.get('gpu_torch'):
        gpu = bert_gpu['gpu_torch']
        if bert_gpu.get('gpu_memoria_total_gb'):
            gpu += f", {bert_gpu['gpu_memoria_total_gb']:.0f} GB"
    partes_hw = []
    # A GPU só entra na legenda se alguma linha da tabela a usou.
    if gpu and (bert_gpu.get('paginas') or ((d.get('llm') or {}).get('modos'))):
        partes_hw.append(f'GPU: {_tex(gpu)}')
    if hw.get('cpu'):
        partes_hw.append(f"CPU: {_tex(hw['cpu'])}")
    if hw.get('memoria_ram_gb'):
        partes_hw.append(f"{hw['memoria_ram_gb']:.0f} GB RAM")

    llm = d.get('llm') or {}
    tem_bert = any(l[0].startswith('BERTimbau') for l in linhas)
    tem_llm = bool(llm.get('modos'))
    reps = []
    if tem_bert:
        reps.append(f"{cfg['repeticoes_bertimbau']} runs for BERTimbau")
    if tem_llm:
        reps.append(f"{cfg['repeticoes_llm']} for the generative model")

    legenda = (f'Inference latency per result page of {videos} videos with $k$ sentences '
               'classified per video, on real sentences from the held-out test set, '
               'measured on the same machine')
    legenda += f" ({'; '.join(partes_hw)})." if partes_hw else '.'
    legenda += (' Wall-clock seconds per page, median and 95th percentile over repeated '
                f"runs after warm-up ({', '.join(reps)}); the last column divides the "
                f'median page time at $k = {k_max}$ by the number of sentences.')
    if tem_bert:
        legenda += ' BERTimbau classifies the whole page in one batch.'
    if tem_llm:
        mem = llm.get('memoria') or {}
        nota_mem = ''
        if mem.get('na_gpu') == 'total':
            nota_mem = ', entirely in GPU memory'
        elif mem.get('na_gpu') == 'parcial' and mem.get('fracao_na_gpu') is not None:
            nota_mem = (f", with {mem['fracao_na_gpu'] * 100:.0f}\\% of the model in GPU "
                        'memory and the rest on the CPU')
        elif mem.get('na_gpu') == 'nao':
            nota_mem = ', on the CPU'
        versao = f" {_tex(llm['versao_ollama'])}" if llm.get('versao_ollama') else ''
        legenda += (f" The generative model (\\texttt{{{_tex(llm.get('modelo'))}}}, served by "
                    f'Ollama{versao}{nota_mem}) receives the few-shot prompt of the '
                    'generative baseline, one request per sentence, sent either '
                    'sequentially or as $c$ concurrent requests; \\emph{free} is the '
                    'decoding of the baseline, and \\emph{constrained} restricts the reply '
                    'to a JSON schema with the three labels.')
    legenda += ' Model loading and transcript retrieval are not included.'

    L = []
    add = L.append
    add('% Gerado por app/nlp/evaluation/scripts/medir_latencia.py')
    add('% NAO EDITAR A MAO: reexecute o script (ou --so-tabela <json>).')
    if fonte:
        add(f'% Fonte: {fonte}')
    add('\\begin{table*}[!t]')
    add('\\centering')
    add('\\caption{' + legenda + '}')
    add('\\label{tab:latencia}')
    add('\\begin{tabular}{ll' + 'cc' * len(ks) + 'c}')
    add('\\toprule')
    cab = ' & '
    for k in ks:
        n = int(k) * videos
        cab += f' & \\multicolumn{{2}}{{c}}{{$k = {k}$ ({n} sentences)}}'
    add(cab + ' & Per sentence \\\\')
    add(''.join(f'\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}' for i in range(len(ks))))
    add('Model & Setting' + ' & Median & p95' * len(ks) + f' & at $k = {k_max}$ \\\\')
    add(' & ' + ' & (s) & (s)' * len(ks) + ' & (ms) \\\\')
    add('\\midrule')
    for modelo, rotulo, paginas in linhas:
        cel = [modelo, rotulo]
        for k in ks:
            p = paginas.get(k) or {}
            cel += [_seg(p.get('mediana_s')), _seg(p.get('p95_s'))]
        cel.append(_ms((paginas.get(k_max) or {}).get('por_frase_mediana_ms')))
        add(' & '.join(cel) + ' \\\\')
    add('\\bottomrule')
    add('\\end{tabular}')
    add('\\end{table*}')

    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text('\n'.join(L) + '\n', encoding='utf-8')
    print(f'  Tabela: {destino}')
    return destino


# ---------------------------------------------------------------------------

def _args():
    ap = argparse.ArgumentParser(description='Latência por página: BERTimbau x LLM')
    ap.add_argument('--modelo', default=None,
                    help='Pasta do BERTimbau ajustado (padrão: modelo de referência da rodada)')
    ap.add_argument('--ks', type=int, nargs='+', default=KS_PADRAO,
                    help='Frases por vídeo (padrão: 3 12)')
    ap.add_argument('--videos', type=int, default=VIDEOS_POR_PAGINA,
                    help='Vídeos por página (padrão: 10)')
    ap.add_argument('--repeticoes', type=int, default=30,
                    help='Repetições por página do BERTimbau (padrão: 30)')
    ap.add_argument('--aquecimento', type=int, default=3,
                    help='Páginas de aquecimento descartadas, BERTimbau (padrão: 3)')
    ap.add_argument('--lote', type=int, default=0,
                    help='Tamanho do lote do BERTimbau; 0 = a página inteira (padrão)')
    ap.add_argument('--threads', type=int, default=0,
                    help='torch.set_num_threads na CPU; 0 = padrão do PyTorch')
    ap.add_argument('--sem-gpu', action='store_true', help='Não mede o BERTimbau na GPU')
    ap.add_argument('--sem-cpu', action='store_true', help='Não mede o BERTimbau na CPU')
    ap.add_argument('--sem-llm', action='store_true', help='Não mede o LLM (não chama o Ollama)')
    ap.add_argument('--modelo-ollama', default=MODELO_OLLAMA_PADRAO,
                    help=f'Tag do modelo no Ollama (padrão: {MODELO_OLLAMA_PADRAO})')
    ap.add_argument('--modos-llm', nargs='+', default=MODOS_LLM_PADRAO,
                    help='Modos formato:concorrência (padrão: ' + ' '.join(MODOS_LLM_PADRAO) + ')')
    ap.add_argument('--repeticoes-llm', type=int, default=2,
                    help='Repetições por página do LLM (padrão: 2)')
    ap.add_argument('--aquecimento-llm', type=int, default=3,
                    help='Requisições de aquecimento por modo (padrão: 3)')
    ap.add_argument('--num-predict-restrito', type=int, default=NUM_PREDICT_RESTRITO,
                    help=f'num_predict na saída restrita (padrão: {NUM_PREDICT_RESTRITO})')
    ap.add_argument('--timeout-llm', type=int, default=120,
                    help='Timeout por requisição ao Ollama, em segundos (padrão: 120)')
    ap.add_argument('--manter-llm', action='store_true',
                    help='Não descarrega o modelo do Ollama ao fim da parte do LLM')
    ap.add_argument('--saida', default=None,
                    help='Pasta para o JSON e a tabela (padrão: resultados e latex da rodada)')
    ap.add_argument('--so-tabela', default=None, metavar='JSON',
                    help='Só regera tab-latencia.tex a partir de um latencia_*.json')
    a = ap.parse_args()
    for modo in a.modos_llm:
        formato, _, conc = modo.partition(':')
        if formato not in ('livre', 'restrita') or not conc.isdigit() or int(conc) < 1:
            ap.error(f'modo inválido: {modo!r} (use livre:1, restrita:4, ...)')
    return a


def main():
    args = _args()
    if args.sem_gpu:
        # Antes de qualquer uso de CUDA (que é preguiçoso): esconde a GPU.
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    from app.nlp import experimento

    if args.so_tabela:
        with open(args.so_tabela, encoding='utf-8') as f:
            d = json.load(f)
        pasta = Path(args.saida) if args.saida else experimento.dir_latex('tabelas')
        escrever_tabela(d, pasta / 'tab-latencia.tex', fonte=Path(args.so_tabela).name)
        return

    ks = sorted(set(args.ks))
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    print('=' * 78)
    print(f'LATÊNCIA POR PÁGINA ({args.videos} vídeos x k frases; k = {ks})')
    print('=' * 78, flush=True)

    paginas, frases_info, (X_train, y_train) = montar_paginas(ks, args.videos, SEMENTE)
    saida = {
        'timestamp': ts,
        'script': 'app/nlp/evaluation/scripts/medir_latencia.py',
        'experimento': experimento.descricao(),
        'configuracao': {
            'videos_por_pagina': args.videos, 'ks': ks, 'semente': SEMENTE,
            'repeticoes_bertimbau': args.repeticoes, 'aquecimento_bertimbau': args.aquecimento,
            'lote_bertimbau': args.lote or 'página inteira',
            'repeticoes_llm': args.repeticoes_llm, 'aquecimento_llm': args.aquecimento_llm,
            'modos_llm': args.modos_llm,
            'medida': 'tempo de parede por página (time.perf_counter); carga do modelo à parte',
            'percentil': 'interpolação linear (regra do numpy.percentile)',
        },
        'frases': frases_info,
        'hardware': info_hardware(),
        'recuperacao_transcricao': RECUPERACAO_TRANSCRICAO,
        'orcamento_latencia': ORCAMENTO_LATENCIA,
    }

    # (b) LLM primeiro: nenhum contexto CUDA deste processo disputa VRAM com o Ollama.
    if args.sem_llm:
        saida['llm'] = {'pulado': '--sem-llm'}
    else:
        print('\n--- LLM (Ollama) ---', flush=True)
        from app.nlp.training import train_toxicity_llm_baselines as base
        fewshot = base.montar_fewshot(X_train, y_train)
        saida['llm'] = medir_llm(args.modelo_ollama, paginas, fewshot, args.modos_llm,
                                 args.repeticoes_llm, args.aquecimento_llm,
                                 args.num_predict_restrito, args.timeout_llm, args.manter_llm)

    # (a) BERTimbau
    caminho = Path(args.modelo) if args.modelo else experimento.caminho_modelo_referencia()
    saida['bertimbau'] = {'modelo': _caminho_publico(caminho)}
    import torch
    if args.sem_gpu:
        saida['bertimbau']['gpu'] = {'pulado': '--sem-gpu'}
    elif not torch.cuda.is_available():
        saida['bertimbau']['gpu'] = {'pulado': 'CUDA indisponível'}
    else:
        print('\n--- BERTimbau na GPU ---', flush=True)
        saida['bertimbau']['gpu'] = medir_bertimbau(caminho, 'cuda', paginas, args.repeticoes,
                                                    args.aquecimento, args.lote, args.threads)
    if args.sem_cpu:
        saida['bertimbau']['cpu'] = {'pulado': '--sem-cpu'}
    else:
        print('\n--- BERTimbau na CPU ---', flush=True)
        saida['bertimbau']['cpu'] = medir_bertimbau(caminho, 'cpu', paginas, args.repeticoes,
                                                    args.aquecimento, args.lote, args.threads)

    saida['versoes'] = versoes_bibliotecas()
    if saida['versoes'].get('torch') and not args.sem_gpu and torch.cuda.is_available():
        saida['versoes']['cudnn'] = torch.backends.cudnn.version()
    if isinstance(saida.get('llm'), dict) and saida['llm'].get('versao_ollama'):
        saida['versoes']['ollama'] = saida['llm']['versao_ollama']

    pasta_json = Path(args.saida) if args.saida else experimento.dir_resultados()
    pasta_json.mkdir(parents=True, exist_ok=True)
    arq = pasta_json / f'latencia_{ts}.json'
    arq.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nJSON: {arq}')

    pasta_tex = Path(args.saida) if args.saida else experimento.dir_latex('tabelas')
    escrever_tabela(saida, pasta_tex / 'tab-latencia.tex', fonte=arq.name)


if __name__ == '__main__':
    main()
