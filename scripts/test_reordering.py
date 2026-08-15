"""
Teste de reordenação do YouTube Safe Kids (atende à Revisão 1 — item F1).
--------------------------------------------------------------------------
Objetivo: para um conjunto de termos de busca, verificar se o filtro de
Toxicidade efetivamente ALTERA a ordem da listagem devolvida pela API do
YouTube e QUANTIFICAR essa alteração — distinguindo:

  (a) "apenas reordenação": os mesmos vídeos, em ordem diferente;
  (b) "mudança de classe": vídeos que saem da faixa "Seguro" por causa da
      toxicidade detectada (banda de score / classe predita).

Cada busca é limitada a MAX_RESULTS=5 vídeos (janela de 5), conforme o
planejamento em documentos/planejamento-teste-reordenacao.md.

IMPORTANTE — consumo de cota da API do YouTube:
  - search().list()  = 100 unidades por busca
  - videos().list()  =   1 unidade por busca
  - youtube-transcript-api = 0 unidades (scraping)
  => ~101 unidades/busca x 10 buscas ~= 1.010 unidades (cota diária: 10.000).

Uso:
    python src/scripts/test_reordering.py             # roda os 10 termos
    python src/scripts/test_reordering.py --limit 3   # roda só os 3 primeiros
    python src/scripts/test_reordering.py --terms "Bob Esponja" "South Park"

Saídas (em src/scripts/reordering_results/):
    reordering_<timestamp>.json   — resumo por busca + agregados
    reordering_<timestamp>.csv     — uma linha por vídeo (ordem original vs nova)
"""

import os
import sys
import csv
import json
import asyncio
import argparse
import logging
from datetime import datetime

# Setup do path do projeto (src/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from app.core.youtube import YouTubeAPI
from app.core.config import get_settings
from app.filters.toxicity import ToxicityFilter

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ======================================================================
# Configuração do teste
# ======================================================================

MAX_RESULTS = 5

# 10 termos de busca — mistura de conteúdo tipicamente seguro (canais
# infantis) e animações de apelo visual infantil porém linguagem adulta,
# para exercitar cenários com e sem reordenação. Ver justificativa no
# documento de planejamento.
SEARCH_TERMS = [
    # --- Tipicamente seguros (esperado: pouca ou nenhuma reordenação) ---
    "Bob Esponja português",
    "Peppa Pig português",
    "Galinha Pintadinha",
    "Turma da Monica desenho",
    "Show da Luna episodios",
    "Patati Patata musicas",
    # --- Apelo infantil + risco (esperado: reordenação/penalização) ---
    "Pernalonga desenho português",
    "South Park dublado português",
    "The Boondocks dublado",
    "Happy Tree Friends português",
]

# Bandas de score de segurança (coerentes com o indicador cromático do protótipo)
BANDA_SEGURO = 0.70     # score >= 0.70  -> Seguro (verde)
BANDA_ATENCAO = 0.40    # 0.40 <= score < 0.70 -> Atenção (amarelo); < 0.40 -> Risco (vermelho)

CLASS_NAMES = ['Nenhuma', 'Leve', 'Severa']


def banda_de_score(score: float) -> str:
    if score >= BANDA_SEGURO:
        return 'Seguro'
    if score >= BANDA_ATENCAO:
        return 'Atencao'
    return 'Risco'


def kendall_tau_distance(order_a, order_b) -> int:
    """Número de pares discordantes entre duas ordenações dos mesmos itens."""
    index_b = {item: i for i, item in enumerate(order_b)}
    n = len(order_a)
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_i, a_j = order_a[i], order_a[j]
            if a_i in index_b and a_j in index_b:
                if index_b[a_i] > index_b[a_j]:
                    discordant += 1
    return discordant


def classe_do_video(tox_filter: ToxicityFilter, video: dict) -> int:
    """
    Classe de toxicidade do vídeo = pior (maior) classe predita entre as
    3 frases representativas (início, meio, fim). 0=Nenhuma, 1=Leve, 2=Severa.
    """
    sentences = video.get('sentences', {})
    worst = 0
    for key in ('start', 'middle', 'end'):
        text = (sentences.get(key) or '').strip()
        if not text:
            continue
        try:
            pred = tox_filter.model.predict_toxicity(text, return_probabilities=True)
            worst = max(worst, int(pred.get('predicted_class', 0)))
        except Exception as e:  # pragma: no cover
            logger.warning(f"Falha ao classificar frase: {e}")
    return worst


async def avaliar_busca(yt: YouTubeAPI, tox: ToxicityFilter, termo: str) -> dict:
    """Executa uma busca e compara a ordem original (YouTube) com a reordenada."""
    print(f"\n{'='*70}\nBusca: '{termo}'\n{'='*70}")

    videos = await yt.search_videos(
        termo, max_results=MAX_RESULTS, nlp_filters_enabled=["Toxicidade"]
    )

    if not videos:
        print("  (nenhum vídeo com transcrição retornado)")
        return {'termo': termo, 'n_videos': 0, 'videos': []}

    # Ordem original = ordem de relevância devolvida pela API do YouTube
    original_order = [v['id'] for v in videos]

    itens = []
    for pos, v in enumerate(videos):
        score = tox.process(v)                    # score de segurança (produção)
        classe = classe_do_video(tox, v)          # pior classe entre as 3 frases
        itens.append({
            'id': v['id'],
            'title': v.get('title', ''),
            'pos_original': pos,
            'score': round(float(score), 4),
            'classe': CLASS_NAMES[classe],
            'classe_idx': classe,
            'banda': banda_de_score(score),
        })

    # Reordena por score decrescente (mesma lógica do endpoint /search)
    reordered = sorted(itens, key=lambda x: x['score'], reverse=True)
    new_order = [it['id'] for it in reordered]
    for nova_pos, it in enumerate(reordered):
        it['pos_nova'] = nova_pos
        it['delta'] = it['pos_original'] - nova_pos  # >0 subiu, <0 desceu

    houve_reordenacao = original_order != new_order
    n_movidos = sum(1 for it in itens if it['pos_original'] != it['pos_nova'])
    max_desloc = max((abs(it['delta']) for it in itens), default=0)
    tau = kendall_tau_distance(original_order, new_order)
    top1_mudou = original_order[0] != new_order[0]
    n_nao_seguro = sum(1 for it in itens if it['classe_idx'] > 0)
    bandas = {b: sum(1 for it in itens if it['banda'] == b)
              for b in ('Seguro', 'Atencao', 'Risco')}

    # Log legível
    print(f"  {'#orig':>5} {'#novo':>5} {'score':>6} {'classe':>8} {'banda':>8}  título")
    for it in reordered:
        print(f"  {it['pos_original']:>5} {it['pos_nova']:>5} {it['score']:>6.2f} "
              f"{it['classe']:>8} {it['banda']:>8}  {it['title'][:45]}")
    print(f"  -> reordenou={houve_reordenacao} | movidos={n_movidos}/{len(itens)} | "
          f"tau={tau} | top1_mudou={top1_mudou} | nao_seguros={n_nao_seguro}")

    return {
        'termo': termo,
        'n_videos': len(itens),
        'houve_reordenacao': houve_reordenacao,
        'n_movidos': n_movidos,
        'max_deslocamento': max_desloc,
        'kendall_tau_distance': tau,
        'top1_mudou': top1_mudou,
        'n_nao_seguro': n_nao_seguro,
        'distribuicao_bandas': bandas,
        'videos': itens,
    }


async def main_async(termos, limit):
    settings = get_settings()
    if not settings.YOUTUBE_API_KEY:
        print("ERRO: YOUTUBE_API_KEY não configurada (.env). Abortando.")
        return

    yt = YouTubeAPI(settings.YOUTUBE_API_KEY)
    tox = ToxicityFilter()
    if tox.model is None:
        print("ERRO: modelo de toxicidade não carregado (treine antes). Abortando.")
        return

    if limit:
        termos = termos[:limit]

    resumo = []
    for termo in termos:
        try:
            resumo.append(await avaliar_busca(yt, tox, termo))
        except Exception as e:
            logger.error(f"Erro na busca '{termo}': {e}")
            resumo.append({'termo': termo, 'erro': str(e)})

    # ---------------- Agregados ----------------
    validos = [r for r in resumo if r.get('n_videos', 0) > 0]
    n_buscas_reordenadas = sum(1 for r in validos if r.get('houve_reordenacao'))
    total_videos = sum(r['n_videos'] for r in validos)
    total_nao_seguros = sum(r.get('n_nao_seguro', 0) for r in validos)
    total_movidos = sum(r.get('n_movidos', 0) for r in validos)

    print(f"\n{'#'*70}\nRESUMO AGREGADO\n{'#'*70}")
    print(f"Buscas avaliadas (com vídeos): {len(validos)}/{len(termos)}")
    print(f"Buscas em que houve reordenação: {n_buscas_reordenadas}/{len(validos)}")
    print(f"Vídeos avaliados no total: {total_videos}")
    print(f"Vídeos que mudaram de posição: {total_movidos}/{total_videos}")
    print(f"Vídeos fora da classe 'Nenhuma': {total_nao_seguros}/{total_videos}")

    agregado = {
        'buscas_avaliadas': len(validos),
        'buscas_totais': len(termos),
        'buscas_reordenadas': n_buscas_reordenadas,
        'total_videos': total_videos,
        'total_videos_movidos': total_movidos,
        'total_videos_nao_seguros': total_nao_seguros,
    }

    # ---------------- Persistência ----------------
    out_dir = os.path.join(os.path.dirname(__file__), 'reordering_results')
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(out_dir, f"reordering_{ts}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'config': {'max_results': MAX_RESULTS, 'bandas': [BANDA_SEGURO, BANDA_ATENCAO]},
                   'agregado': agregado, 'buscas': resumo}, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(out_dir, f"reordering_{ts}.csv")
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['termo', 'pos_original', 'pos_nova', 'delta', 'score', 'classe', 'banda', 'id', 'title'])
        for r in validos:
            for it in sorted(r['videos'], key=lambda x: x['pos_original']):
                w.writerow([r['termo'], it['pos_original'], it['pos_nova'], it['delta'],
                            it['score'], it['classe'], it['banda'], it['id'], it['title']])

    print(f"\nResultados salvos em:\n  {json_path}\n  {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Teste de reordenação do YouTube Safe Kids")
    parser.add_argument('--limit', type=int, default=None, help="Roda apenas os N primeiros termos")
    parser.add_argument('--terms', nargs='*', default=None, help="Sobrescreve a lista de termos")
    args = parser.parse_args()
    termos = args.terms if args.terms else SEARCH_TERMS
    asyncio.run(main_async(termos, args.limit))


if __name__ == "__main__":
    main()
