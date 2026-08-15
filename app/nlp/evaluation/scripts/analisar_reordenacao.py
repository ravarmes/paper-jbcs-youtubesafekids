"""
Compara a ordem de resultados do YouTube com a reordenação pelo score de segurança.

Responde à pergunta: o sistema apenas confirma o ranqueamento do YouTube, ou
produz uma ordem substancialmente diferente? Sem essa medida, não se sabe se o
reclassificador acrescenta algo ao que a plataforma já entrega.

Para cada termo de busca:
  1. consulta a API do YouTube e preserva a ordem original (posições 1..N);
  2. extrai três sentenças representativas da transcrição (início, meio, fim),
     além de título e descrição;
  3. calcula o score de toxicidade e reordena;
  4. compara as duas ordens.

Métricas reportadas por busca e no agregado:
  - correlação de postos de Spearman e tau de Kendall entre as duas ordens;
  - deslocamento médio e máximo de posição;
  - sobreposição do Top-3 e do Top-5 (a "janela" pedida na revisão);
  - mudança de faixa de risco (alta/média/baixa segurança).

Dois scores são calculados: `score_frases`, definido no artigo como a média das
três sentenças, e `score_completo`, que inclui também título e descrição. O
primeiro é o usado na análise principal, para não divergir da Equação do artigo;
o segundo permite avaliar se os metadados alterariam as conclusões.

Uso:
    python -m app.nlp.evaluation.scripts.analisar_reordenacao
    python -m app.nlp.evaluation.scripts.analisar_reordenacao --top-n 10 --termos "desenho infantil" "musica infantil"
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('googleapiclient').setLevel(logging.ERROR)
logging.getLogger('app.core.youtube').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Pesos de penalização — os mesmos do filtro em produção
PESOS = {'Nenhuma': 0.0, 'Leve': 0.5, 'Severa': 1.0}

# Consultas que um responsável plausivelmente digitaria ao buscar conteúdo infantil
TERMOS_PADRAO = [
    'desenho animado infantil',
    'música infantil para crianças',
    'vídeo educativo para crianças',
    'história infantil narrada',
    'desenho pernalonga português',
    'turma da mônica desenho',
    'aprender os números para crianças',
    'desenho animado engraçado',
    'vídeo de brinquedos para crianças',
    'canal infantil desenho em português',
]

# Faixas de risco usadas para verificar mudança de classe, e não só de posição
FAIXAS = [(0.80, 'alta segurança'), (0.50, 'segurança média'), (0.0, 'baixa segurança')]

# A obtenção de transcrições é raspagem, não API oficial: sem espaçamento entre
# as chamadas o YouTube bloqueia o IP depois de algumas dezenas de requisições.
ESPERA_ENTRE_VIDEOS = (18.0, 26.0)  # segundos, sorteados no intervalo
ESPERA_APOS_BLOQUEIO = 300         # segundos antes de tentar de novo
MAX_TENTATIVAS_BLOQUEIO = 4

# As transcrições são estáveis; guardá-las evita repetir requisições a cada
# reexecução da análise — o que também reduz o risco de novo bloqueio.
CACHE_PATH = Path(project_root) / 'app' / 'cache' / 'transcricoes_reordenacao.json'


def carregar_cache():
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding='utf-8') as f:
            return json.load(f)
    return {}


def salvar_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)


async def frases_com_cache(api, video_id, cache):
    """
    Busca as três sentenças do vídeo, servindo do cache quando possível e
    recuando diante de bloqueio de IP.

    Distingue-se o vídeo sem legenda (resultado legítimo, cacheado) do bloqueio
    de IP (falha transitória, não cacheada) — confundir os dois registraria como
    "sem transcrição" vídeos que apenas não puderam ser consultados.
    """
    if video_id in cache:
        return cache[video_id], True

    for tentativa in range(1, MAX_TENTATIVAS_BLOQUEIO + 1):
        try:
            frases = await api.get_video_sentences(video_id)
            cache[video_id] = frases
            return frases, False
        except Exception as e:
            nome = type(e).__name__
            if 'IpBlocked' in nome or 'RequestBlocked' in nome or 'TooManyRequests' in nome:
                logger.warning(f'  IP bloqueado; aguardando {ESPERA_APOS_BLOQUEIO}s '
                               f'(tentativa {tentativa}/{MAX_TENTATIVAS_BLOQUEIO})')
                time.sleep(ESPERA_APOS_BLOQUEIO)
                continue
            logger.warning(f'  {video_id}: {nome}')
            cache[video_id] = {'start': '', 'middle': '', 'end': ''}
            return cache[video_id], False

    logger.error(f'  {video_id}: bloqueio persistente, vídeo não consultado')
    return None, False


def faixa_de(score):
    for limite, nome in FAIXAS:
        if score >= limite:
            return nome
    return FAIXAS[-1][1]


def carregar_modelo():
    from app.nlp.models.bertimbau_toxicity import BertimbauToxicity
    trained = Path(project_root) / 'app' / 'nlp' / 'models' / 'trained'
    candidatos = [p for p in trained.glob('TOX_*') if (p / 'config.json').exists()]
    if not candidatos:
        raise FileNotFoundError(f'Nenhum modelo TOX_* treinado em {trained}')
    caminho = max(candidatos, key=lambda p: p.stat().st_mtime)
    logger.info(f'Modelo: {caminho.name}')
    return BertimbauToxicity(model_path=str(caminho))


def score_de(modelo, textos):
    """
    Score de segurança de um conjunto de textos: média de $1 - \\sum p_i w_i$.

    Textos vazios são ignorados; se nenhum sobrar, devolve None, para que o
    vídeo possa ser tratado à parte em vez de receber um valor arbitrário.
    """
    validos = [t for t in textos if t and t.strip()]
    if not validos:
        return None
    scores = []
    for texto in validos:
        probs = modelo.predict_toxicity(texto, return_probabilities=True)['probabilities']
        scores.append(1.0 - sum(probs[c] * PESOS[c] for c in PESOS))
    return float(np.mean(scores))


async def coletar_busca(api, termo, top_n):
    """
    Executa a busca e devolve os vídeos na ordem em que o YouTube os retornou.

    A ordem importa: `videos().list` não preserva a ordem da consulta, então os
    detalhes são reindexados pelos IDs na sequência do `search().list`.
    """
    busca = api.youtube.search().list(
        q=termo, part='snippet', maxResults=top_n, type='video',
        relevanceLanguage='pt', safeSearch='strict'
    ).execute()

    ids = [i['id']['videoId'] for i in busca.get('items', [])]
    if not ids:
        return []

    detalhes = api.youtube.videos().list(
        part='snippet,contentDetails,statistics', id=','.join(ids)
    ).execute()
    por_id = {i['id']: i for i in detalhes.get('items', [])}

    videos = []
    for posicao, vid in enumerate(ids, 1):
        item = por_id.get(vid)
        if not item:
            continue
        videos.append({
            'posicao_youtube': posicao,
            'id': vid,
            'titulo': item['snippet']['title'],
            'descricao': item['snippet'].get('description', ''),
        })
    return videos


async def analisar_termo(api, modelo, termo, top_n, cache):
    logger.info(f'\n=== "{termo}"')
    videos = await coletar_busca(api, termo, top_n)
    if not videos:
        logger.warning('  nenhum vídeo retornado')
        return None

    for v in videos:
        frases, veio_do_cache = await frases_com_cache(api, v['id'], cache)
        if frases is None:
            v['nao_consultado'] = True
            frases = {}
        if not veio_do_cache:
            time.sleep(random.uniform(*ESPERA_ENTRE_VIDEOS))

        v['frases'] = {k: frases.get(k, '') for k in ('start', 'middle', 'end')}
        v['tem_transcricao'] = any(t.strip() for t in v['frases'].values())

        v['score_frases'] = score_de(modelo, list(v['frases'].values()))
        v['score_completo'] = score_de(
            modelo, [v['titulo'], v['descricao'][:500]] + list(v['frases'].values())
        )
        marca = '' if v['tem_transcricao'] else '  (sem transcrição)'
        s = v['score_frases']
        logger.info(f"  {v['posicao_youtube']:>2}. score={'—' if s is None else f'{s:.3f}'}"
                    f"{marca}  {v['titulo'][:58]}")

    return {'termo': termo, 'videos': videos}


def comparar_ordens(videos, chave='score_frases'):
    """
    Compara a ordem do YouTube com a ordem induzida pelo score.

    Só entram vídeos com score definido; os sem transcrição são contados à
    parte, porque incluí-los exigiria arbitrar um valor e isso enviesaria a
    comparação.
    """
    from scipy.stats import kendalltau, spearmanr

    com_score = [v for v in videos if v.get(chave) is not None]
    if len(com_score) < 3:
        return None

    # Posição do YouTube entre os vídeos pontuáveis, renumerada de 1 a N
    ordem_yt = sorted(com_score, key=lambda v: v['posicao_youtube'])
    for i, v in enumerate(ordem_yt, 1):
        v['_rank_yt'] = i

    # Ordem do modelo: score decrescente (mais seguro primeiro)
    ordem_modelo = sorted(com_score, key=lambda v: -v[chave])
    for i, v in enumerate(ordem_modelo, 1):
        v['_rank_modelo'] = i

    yt = [v['_rank_yt'] for v in ordem_yt]
    mo = [v['_rank_modelo'] for v in ordem_yt]

    desloc = [abs(v['_rank_yt'] - v['_rank_modelo']) for v in ordem_yt]

    def sobreposicao(k):
        if len(com_score) < k:
            return None
        topo_yt = {v['id'] for v in ordem_yt[:k]}
        topo_mo = {v['id'] for v in ordem_modelo[:k]}
        return len(topo_yt & topo_mo) / k

    rho, p_rho = spearmanr(yt, mo)
    tau, p_tau = kendalltau(yt, mo)

    return {
        'n_videos_pontuados': len(com_score),
        'n_sem_transcricao': len(videos) - len(com_score),
        'spearman': float(rho), 'spearman_p': float(p_rho),
        'kendall': float(tau), 'kendall_p': float(p_tau),
        'deslocamento_medio': float(np.mean(desloc)),
        'deslocamento_maximo': int(np.max(desloc)),
        'inalterados': int(sum(1 for d in desloc if d == 0)),
        'sobreposicao_top3': sobreposicao(3),
        'sobreposicao_top5': sobreposicao(5),
        'primeiro_do_youtube_permanece': bool(ordem_yt[0]['id'] == ordem_modelo[0]['id']),
        'faixas': {
            nome: sum(1 for v in com_score if faixa_de(v[chave]) == nome)
            for _, nome in FAIXAS
        },
    }


async def main_async(termos, top_n):
    from app.core.config import get_settings
    from app.core.youtube import YouTubeAPI

    settings = get_settings()
    api = YouTubeAPI(settings.YOUTUBE_API_KEY)
    modelo = carregar_modelo()

    cache = carregar_cache()
    logger.info(f'Cache de transcrições: {len(cache)} vídeos já conhecidos')

    buscas = []
    try:
        for termo in termos:
            resultado = await analisar_termo(api, modelo, termo, top_n, cache)
            if not resultado:
                continue
            resultado['comparacao_frases'] = comparar_ordens(resultado['videos'], 'score_frases')
            resultado['comparacao_completo'] = comparar_ordens(resultado['videos'], 'score_completo')
            buscas.append(resultado)
            salvar_cache(cache)   # preserva o progresso a cada busca
    finally:
        salvar_cache(cache)

    return buscas


def resumo(buscas, chave='comparacao_frases'):
    """Agrega as métricas de todas as buscas."""
    comps = [b[chave] for b in buscas if b.get(chave)]
    if not comps:
        return None

    def media(campo):
        vals = [c[campo] for c in comps if c.get(campo) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        'n_buscas': len(comps),
        'n_videos': sum(c['n_videos_pontuados'] for c in comps),
        'n_sem_transcricao': sum(c['n_sem_transcricao'] for c in comps),
        'spearman_medio': media('spearman'),
        'kendall_medio': media('kendall'),
        'deslocamento_medio': media('deslocamento_medio'),
        'deslocamento_maximo': max(c['deslocamento_maximo'] for c in comps),
        'pct_posicoes_inalteradas': float(
            sum(c['inalterados'] for c in comps) / sum(c['n_videos_pontuados'] for c in comps)
        ),
        'sobreposicao_top3_media': media('sobreposicao_top3'),
        'sobreposicao_top5_media': media('sobreposicao_top5'),
        'buscas_com_novo_primeiro': sum(1 for c in comps if not c['primeiro_do_youtube_permanece']),
    }


def carregar_planilha(caminho):
    """
    Lê as buscas de uma planilha preenchida manualmente.

    Colunas esperadas (separador ';', UTF-8):
      TERMO, POSICAO, VIDEO_ID, TITULO, DESCRICAO,
      FRASE_INICIO, FRASE_MEIO, FRASE_FIM

    A coleta manual existe porque a obtenção automática de transcrições é
    raspagem e o YouTube a bloqueia por taxa. Linhas com as três frases vazias
    representam vídeos sem transcrição disponível e são preservadas: a fração
    delas é, por si, um resultado sobre a cobertura do método.
    """
    import pandas as pd

    df = pd.read_csv(caminho, sep=None, engine='python', encoding='utf-8-sig',
                     dtype=str, keep_default_na=False)
    faltando = {'TERMO', 'POSICAO', 'FRASE_INICIO', 'FRASE_MEIO', 'FRASE_FIM'} - set(df.columns)
    if faltando:
        raise ValueError(f'Colunas ausentes na planilha: {sorted(faltando)}')

    df = df[df['TERMO'].str.strip() != '']
    df['POSICAO'] = df['POSICAO'].astype(int)

    buscas = []
    for termo, grupo in df.groupby('TERMO', sort=False):
        estrato = grupo['ESTRATO'].iloc[0].strip() if 'ESTRATO' in grupo.columns else 'unico'
        videos = []
        for _, linha in grupo.sort_values('POSICAO').iterrows():
            frases = {
                'start': linha.get('FRASE_INICIO', '').strip(),
                'middle': linha.get('FRASE_MEIO', '').strip(),
                'end': linha.get('FRASE_FIM', '').strip(),
            }
            videos.append({
                'posicao_youtube': int(linha['POSICAO']),
                'id': linha.get('VIDEO_ID', '').strip() or f"{termo}#{linha['POSICAO']}",
                'titulo': linha.get('TITULO', '').strip(),
                'descricao': linha.get('DESCRICAO', '').strip(),
                'frases': frases,
                'tem_transcricao': any(t for t in frases.values()),
            })
        buscas.append({'termo': termo, 'estrato': estrato, 'videos': videos})
    return buscas


def pontuar_buscas(buscas, modelo):
    """Calcula os dois scores de cada vídeo e as comparações de ordem."""
    for busca in buscas:
        logger.info(f'\n=== "{busca["termo"]}"')
        for v in busca['videos']:
            v['score_frases'] = score_de(modelo, list(v['frases'].values()))
            v['score_completo'] = score_de(
                modelo, [v['titulo'], v['descricao'][:500]] + list(v['frases'].values())
            )
            s = v['score_frases']
            marca = '' if v['tem_transcricao'] else '  (sem transcrição)'
            logger.info(f"  {v['posicao_youtube']:>2}. "
                        f"score={'—' if s is None else f'{s:.3f}'}{marca}  {v['titulo'][:56]}")
        busca['comparacao_frases'] = comparar_ordens(busca['videos'], 'score_frases')
        busca['comparacao_completo'] = comparar_ordens(busca['videos'], 'score_completo')
    return buscas


def main():
    parser = argparse.ArgumentParser(description='Compara ordem do YouTube vs. reordenação')
    parser.add_argument('--termos', nargs='*', default=None)
    parser.add_argument('--top-n', type=int, default=10)
    parser.add_argument('--csv', default=None,
                        help='Planilha já preenchida; dispensa a API do YouTube')
    args = parser.parse_args()

    if args.csv:
        print(f'Lendo buscas de {args.csv}\n')
        buscas = pontuar_buscas(carregar_planilha(args.csv), carregar_modelo())
    else:
        termos = args.termos or TERMOS_PADRAO
        print(f'Analisando {len(termos)} buscas, Top-{args.top_n} de cada\n')
        buscas = asyncio.run(main_async(termos, args.top_n))
    if not buscas:
        print('Nenhuma busca retornou resultados.')
        return

    agregado = resumo(buscas, 'comparacao_frases')
    agregado_completo = resumo(buscas, 'comparacao_completo')

    # Os estratos medem coisas diferentes e não devem ser fundidos numa média:
    # no A o desejável é reordenar pouco (não atrapalhar quando não há risco);
    # no B, reordenar muito (proteger quando há).
    por_estrato = {}
    for estrato in dict.fromkeys(b.get('estrato', 'unico') for b in buscas):
        subset = [b for b in buscas if b.get('estrato', 'unico') == estrato]
        por_estrato[estrato] = resumo(subset, 'comparacao_frases')

    print('\n' + '=' * 84)
    print('REORDENAÇÃO POR BUSCA (score das três sentenças)')
    print('=' * 84)
    print(f"{'Estrato':<16} {'Termo':<32} {'n':>3} {'Spearman':>9} {'Desloc.':>8} "
          f"{'Top-3':>7} {'Top-5':>7}")
    print('-' * 84)
    for b in buscas:
        c = b.get('comparacao_frases')
        if not c:
            print(f"{b.get('estrato','')[:16]:<16} {b['termo'][:32]:<32} {'—':>3}  "
                  f"(poucos vídeos com transcrição)")
            continue
        t3 = '—' if c['sobreposicao_top3'] is None else f"{c['sobreposicao_top3']:.0%}"
        t5 = '—' if c['sobreposicao_top5'] is None else f"{c['sobreposicao_top5']:.0%}"
        print(f"{b.get('estrato','')[:16]:<16} {b['termo'][:32]:<32} "
              f"{c['n_videos_pontuados']:>3} {c['spearman']:>9.3f} "
              f"{c['deslocamento_medio']:>8.2f} {t3:>7} {t5:>7}")
    print('=' * 84)

    if agregado:
        print('\nAGREGADO')
        print(f"  vídeos pontuados: {agregado['n_videos']} "
              f"(+{agregado['n_sem_transcricao']} sem transcrição)")
        print(f"  correlação de Spearman média: {agregado['spearman_medio']:+.3f}")
        print(f"  tau de Kendall médio:         {agregado['kendall_medio']:+.3f}")
        print(f"  deslocamento médio de posição: {agregado['deslocamento_medio']:.2f}")
        print(f"  posições inalteradas: {agregado['pct_posicoes_inalteradas']:.1%}")
        print(f"  sobreposição Top-3: {agregado['sobreposicao_top3_media']:.1%} | "
              f"Top-5: {agregado['sobreposicao_top5_media']:.1%}")
        print(f"  buscas em que o 1º do YouTube deixou de ser o 1º: "
              f"{agregado['buscas_com_novo_primeiro']} de {agregado['n_buscas']}")

    if agregado_completo:
        print(f"\n  [com título e descrição] Spearman médio: "
              f"{agregado_completo['spearman_medio']:+.3f} | "
              f"deslocamento: {agregado_completo['deslocamento_medio']:.2f}")

    destino = Path(project_root) / 'app' / 'nlp' / 'evaluation' / 'results' / \
        f'reordenacao_{datetime.now():%Y%m%d_%H%M%S}.json'
    with open(destino, 'w', encoding='utf-8') as f:
        json.dump({'timestamp': datetime.now().isoformat(), 'top_n': args.top_n,
                   'origem': args.csv or 'API do YouTube',
                   'termos': [b['termo'] for b in buscas], 'buscas': buscas,
                   'agregado_frases': agregado, 'agregado_completo': agregado_completo,
                   'agregado_por_estrato': por_estrato},
                  f, indent=2, ensure_ascii=False, default=str)
    print(f'\nSalvo em: {destino}')


if __name__ == '__main__':
    main()
