"""
Estudo de caso: o score separa buscas de risco de buscas seguras?

A análise de reordenação (`analisar_reordenacao.py`) mede o quanto a ordem muda
dentro de cada busca. Esta aqui responde à pergunta anterior a essa: o score
distingue conteúdo que uma criança encontraria e não deveria ver daquele que um
responsável procuraria deliberadamente? Os estratos são definidos pela BUSCA, não
por anotação dos vídeos: a AUC mede a separação entre dois estratos de busca, não
a acurácia na detecção de vídeos tóxicos (M4a).

Os dois estratos medem coisas diferentes e não são fundidos numa média:
  A_responsavel    — consultas que um adulto digitaria buscando conteúdo infantil;
                     o desejável é *não* reordenar, porque não há risco a corrigir;
  B_apelo_infantil — animações adultas com apelo visual infantil, que uma criança
                     plausivelmente clicaria; o desejável é rebaixá-las.

O score do vídeo é o do filtro de toxicidade isolado: média dos scores
P = 1 - (0,5*p_leve + 1,0*p_severa) das três frases amostradas (início, meio,
fim). Regra de decisão, explícita em todo o script: score < limiar => rebaixado.

O que reporta (itens do plano de resposta aos revisores, pacote P7):
  - distribuição do score por estrato e por busca (tab-estratos);
  - separação entre estratos: Mann-Whitney e AUC com IC por bootstrap;
  - M4b  limiar DESIGNADO fixado a priori, lido da simulação sobre as predições
         out-of-fold (bloco `limiar_a_priori` do simulacao_amostragem_*.json:
         5% de falso alarme em vídeos sintéticos apropriados, k = 3, média); é o
         resultado principal. Todos os demais limiares são exploratórios;
  - M4c  desfecho em cada limiar da grade (rebaixados em B, falsos alarmes em A),
         com os títulos dos falsos alarmes e das perdas (tab-limiares-estudo-caso);
  - M4d  TODAS as perdas do estrato B no limiar designado, com as três frases,
         classes preditas e scores, e uma categoria (ver classificar_perda). Sem a
         transcrição completa não se afirma que as partes não amostradas contenham
         toxicidade;
  - M1b  quais vídeos do estudo de caso estão no corpus v1 e em que partição, e
         todas as métricas com e sem os vídeos cujas frases estão no treino;
  - M7b  sensibilidade da AUC e do deslocamento ao peso da classe Leve
         (0,25 / 0,5 / 0,75) e à regra de agregação;
  - M7c  deslocamento no estrato A e rebaixamento no B com e sem a banda de
         tolerância (delta da simulação; regra em
         analisar_reordenacao.ordenar_com_banda);
  - m8   figuras fig11/fig12 com o limiar designado e duas cores sem ambiguidade.

Uso:
    python -m app.nlp.evaluation.scripts.analisar_estratos --csv data/reordenacao_buscas.csv
    # sem carregar modelo, reaproveitando as probabilidades de um estratos_*.json:
    python app/nlp/evaluation/scripts/analisar_estratos.py --predicoes <estratos.json> \
        --simulacao <simulacao.json> --saida <pasta>
"""

import argparse
import difflib
import json
import logging
import os
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp import experimento  # noqa: E402

logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PESOS = {'Nenhuma': 0.0, 'Leve': 0.5, 'Severa': 1.0}
N_BOOTSTRAP = 10000
N_BOOT_SENS = 2000
SEED = 42

# Grade de limiares avaliados no estudo de caso. Todos exploratórios, exceto o
# limiar designado (a priori), que é acrescentado à grade.
LIMIARES_GRADE = (0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50)
# Só se não houver simulação nem OOF: limiar da análise de falhas da versão submetida.
LIMIAR_LEGADO = 0.85

PESOS_LEVE_SENS = (0.25, 0.5, 0.75)

# Correspondência com o corpus por texto (para IDs que não são do YouTube)
SIM_TITULO = 0.95
SIM_FRASE = 0.90
# Status que contam como sobreposição nos subconjuntos sem_treino / sem_sobreposicao (M1b)
STATUS_SOBREPOSTO = ('confirmado_por_id', 'provavel_por_titulo_e_frase',
                     'possivel_por_titulo_ou_frase')

NOTA_PARTES_NAO_AMOSTRADAS = (
    'Só três frases de cada vídeo foram coletadas; sem a transcrição completa não se '
    'afirma que as partes não amostradas contenham (ou não) toxicidade.')


def importar_irmao(nome):
    """Importa um script desta pasta sem passar por app.nlp.evaluation (cujo __init__ carrega torch)."""
    import importlib
    pasta = os.path.dirname(os.path.abspath(__file__))
    if pasta not in sys.path:
        sys.path.insert(0, pasta)
    return importlib.import_module(nome)


# =============================================================================
# Pontuação
# =============================================================================

def carregar_modelo(caminho=None):
    from app.nlp.models.bertimbau_toxicity import BertimbauToxicity
    caminho = Path(caminho) if caminho else experimento.caminho_modelo_referencia()
    print(f'Modelo: {caminho.name}\n')
    return BertimbauToxicity(model_path=str(caminho)), caminho


def score_frase(probs, w_leve=PESOS['Leve']):
    """Score de uma frase a partir de [p_nenhuma, p_leve, p_severa]."""
    return 1.0 - (w_leve * probs[1] + PESOS['Severa'] * probs[2])


def classe_de(probs):
    return ('Nenhuma', 'Leve', 'Severa')[int(np.argmax(probs))]


def pontuar(modelo, texto):
    """Probabilidades [p_nenhuma, p_leve, p_severa] de uma frase."""
    probs = modelo.predict_toxicity(texto, return_probabilities=True)['probabilities']
    return [float(probs['Nenhuma']), float(probs['Leve']), float(probs['Severa'])]


def aplicar_probabilidades(v, frases_pontuadas):
    """Preenche os campos de score do vídeo a partir das frases pontuadas."""
    v['frases_pontuadas'] = frases_pontuadas
    v['scores_frases'] = [f['score'] for f in frases_pontuadas]
    v['classes_frases'] = [f['classe'] for f in frases_pontuadas]
    v['score_media'] = float(np.mean(v['scores_frases'])) if frases_pontuadas else None
    v['score_minimo'] = float(np.min(v['scores_frases'])) if frases_pontuadas else None
    v['tem_frase_toxica'] = any(c != 'Nenhuma' for c in v['classes_frases'])


def pontuar_videos(buscas, modelo=None, predicoes=None):
    """
    Pontua as frases de cada vídeo com o modelo, ou reaproveita as probabilidades
    gravadas num estratos_*.json anterior (`predicoes`, chave (termo, posição)).
    """
    mapa = {}
    if predicoes is not None:
        for pv in predicoes.get('videos', []):
            mapa[(pv['termo'], int(pv['posicao_youtube']))] = pv.get('frases_pontuadas', [])
    for b in buscas:
        for v in b['videos']:
            v['estrato'], v['termo'] = b['estrato'], b['termo']
            if predicoes is not None:
                chave = (b['termo'], int(v['posicao_youtube']))
                if chave not in mapa:
                    raise KeyError(f'Vídeo sem predição gravada: {chave}')
                # Rótulo humano gravado num JSON anterior não é reaproveitado: vem sempre
                # do CSV (--rotulos-humanos).
                fp = [dict({k: x for k, x in f.items() if k not in ('humano', 'anotadores')},
                           score=score_frase(f['probs']), classe=classe_de(f['probs']))
                      for f in mapa[chave]]
            else:
                fp = []
                for parte, texto in v['frases'].items():
                    if texto.strip():
                        probs = pontuar(modelo, texto)
                        fp.append({'parte': parte, 'texto': texto, 'probs': probs,
                                   'score': score_frase(probs), 'classe': classe_de(probs)})
            aplicar_probabilidades(v, fp)


# =============================================================================
# Estatística
# =============================================================================

def _auc(sa, sb):
    """AUC com B como positivo e score invertido: P(sB < sA) + 0,5*P(empate)."""
    sa, sb = np.asarray(sa, dtype=float), np.asarray(sb, dtype=float)
    if sa.size == 0 or sb.size == 0:
        return float('nan')
    dif = sa[:, None] - sb[None, :]
    return float(((dif > 0).sum() + 0.5 * (dif == 0).sum()) / dif.size)


def auc_com_ic(scores_a, scores_b, rng, n_boot=N_BOOTSTRAP):
    """
    AUC de separação entre estratos, com IC 95% por bootstrap.

    O estrato B é o "positivo" (deveria ser rebaixado) e o score é invertido, de
    modo que AUC = 1 significa separação perfeita. Reamostra o conjunto unido
    (como na versão submetida) e descarta reamostras com um estrato só; mesma
    sequência de sorteios da versão submetida, que usava roc_auc_score.
    """
    y = np.array([0] * len(scores_a) + [1] * len(scores_b))
    s = np.array(list(scores_a) + list(scores_b), dtype=float)
    ponto = _auc(scores_a, scores_b)
    amostras = []
    for _ in range(n_boot):
        i = rng.integers(0, len(s), len(s))
        yi = y[i]
        if len(np.unique(yi)) < 2:
            continue
        si = s[i]
        amostras.append(_auc(si[yi == 0], si[yi == 1]))
    lo, hi = np.percentile(amostras, [2.5, 97.5])
    return ponto, float(lo), float(hi)


def _resumo_video(v, chave='score_media'):
    return {'id': v['id'], 'titulo': v['titulo'], 'termo': v['termo'],
            'posicao_youtube': int(v['posicao_youtube']), 'score': v[chave]}


def avaliar_limiar(A, B, tau, chave='score_media'):
    """Desfecho num limiar, com a regra explícita score < tau => rebaixado."""
    falsos = sorted([v for v in A if v[chave] < tau], key=lambda v: v[chave])
    perdas = sorted([v for v in B if v[chave] >= tau], key=lambda v: v[chave])
    return {'limiar': float(tau), 'regra': 'score < limiar => rebaixado',
            'detectados_b': len(B) - len(perdas), 'total_b': len(B),
            'falsos_a': len(falsos), 'total_a': len(A),
            'falsos_alarme_a': [_resumo_video(v, chave) for v in falsos],
            'perdas_b': [_resumo_video(v, chave) for v in perdas]}


def separacao(A, B, rng, limiares, chave='score_media', n_boot=N_BOOTSTRAP):
    from scipy.stats import mannwhitneyu

    sa = [v[chave] for v in A]
    sb = [v[chave] for v in B]
    u, p = mannwhitneyu(sa, sb, alternative='greater')
    auc, lo, hi = auc_com_ic(sa, sb, rng, n_boot)
    return {'n_a': len(sa), 'n_b': len(sb),
            'media_a': float(np.mean(sa)), 'media_b': float(np.mean(sb)),
            'diferenca_medias': float(np.mean(sa) - np.mean(sb)),
            'mannwhitney_u': float(u), 'p': float(p), 'auc': auc, 'auc_ic95': [lo, hi],
            'por_limiar': [avaliar_limiar(A, B, t, chave) for t in limiares]}


# =============================================================================
# Limiar designado e banda (da simulação)
# =============================================================================

def obter_limiar_e_banda(pasta_res, arq_simulacao=None):
    """
    Limiar a priori e delta da banda, na ordem de preferência:
      1. bloco limiar_a_priori / banda_tolerancia do simulacao_amostragem_*.json;
      2. recalculados das OOF (mesmos geradores da simulação);
      3. legado: LIMIAR_LEGADO, sem banda (tudo exploratório).
    """
    sim_mod = importar_irmao('simular_amostragem')
    if arq_simulacao:
        with open(arq_simulacao, encoding='utf-8') as fh:
            sim, fonte = json.load(fh), Path(arq_simulacao)
    else:
        sim, fonte = sim_mod.ler_simulacao_mais_recente(pasta_res)

    if sim and (sim.get('limiar_a_priori') or {}).get('valor') is not None:
        la, bt = sim['limiar_a_priori'], sim.get('banda_tolerancia') or {}
        return {
            'valor': float(la['valor']), 'origem': f"simulacao_{la.get('populacao')}",
            'populacao': la.get('populacao'), 'k': la.get('k'), 'regra': la.get('regra'),
            'falso_alarme_alvo_pct': la.get('falso_alarme_alvo_pct'),
            'definicao': la.get('definicao'),
            'ic95_bootstrap_populacao': la.get('ic95_bootstrap_populacao'),
            'aviso': la.get('aviso'), 'fonte': str(fonte),
        }, {
            'delta': bt.get('delta'), 'definicao': bt.get('definicao'),
            'populacao': bt.get('populacao'), 'quantis_diferenca': bt.get('quantis_diferenca'),
            'ic95_bootstrap_populacao': bt.get('ic95_bootstrap_populacao'), 'fonte': str(fonte),
        }

    arquivos = sim_mod.localizar_modelo_final(pasta_res)
    pop = sim_mod.carregar_populacao_oof(arquivos)
    if pop is not None:
        lim, delta, disp = sim_mod.limiar_e_banda(pop)
        fonte = str(arquivos['oof'])
        return {
            'valor': float(lim), 'origem': 'recalculado_oof', 'populacao': 'oof',
            'k': sim_mod.K_SERVICO, 'regra': 'media',
            'falso_alarme_alvo_pct': sim_mod.FALSO_ALARME_ALVO,
            'definicao': 'recalculado das OOF com os geradores da simulação (sem JSON de simulação)',
            'ic95_bootstrap_populacao': None, 'aviso': None, 'fonte': fonte,
        }, {
            'delta': float(delta), 'populacao': 'oof',
            'quantis_diferenca': disp['quantis_diferenca_entre_videos'], 'fonte': fonte,
            'definicao': f'percentil {sim_mod.PERCENTIL_BANDA} de |S1 - S2| (k=3, média, OOF)',
        }

    return {'valor': LIMIAR_LEGADO, 'origem': 'legado', 'populacao': None,
            'definicao': 'sem simulação nem OOF: limiar da análise de falhas da versão '
                         'submetida; nenhum limiar é a priori nesta execução',
            'aviso': 'limiar não fixado a priori'}, {'delta': None}


# =============================================================================
# M1b: sobreposição com o corpus
# =============================================================================

def _norm(texto):
    t = unicodedata.normalize('NFKD', str(texto).lower())
    t = ''.join(c for c in t if not unicodedata.combining(c))
    return ' '.join(re.sub(r'[^a-z0-9 ]+', ' ', t).split())


def _melhor_correspondencia(alvo, candidatos):
    """(similaridade, índice) do candidato mais parecido com `alvo` (difflib)."""
    sm = difflib.SequenceMatcher(None)
    sm.set_seq2(alvo)
    melhor, idx = 0.0, None
    for i, c in enumerate(candidatos):
        sm.set_seq1(c)
        if sm.real_quick_ratio() <= melhor or sm.quick_ratio() <= melhor:
            continue
        r = sm.ratio()
        if r > melhor:
            melhor, idx = r, i
    return melhor, idx


def sobreposicao_corpus(videos):
    """
    Quais vídeos do estudo de caso estão no corpus v1 e em que partição.

    VIDEO_ID com 11 caracteres é ID do YouTube e é comparado com a coluna ID do
    corpus (que coincide com o ID extraído do LINK). Os demais (16 caracteres)
    não são IDs do YouTube -- parecem o parâmetro si= de links de
    compartilhamento -- e, para todos os vídeos, procura-se também correspondência
    por título e por frase (difflib sobre texto normalizado).
    """
    import pandas as pd

    try:
        corpus = pd.read_csv(experimento.DATA_DIR / 'corpus_toxicidade.csv', sep=';',
                             encoding='utf-8-sig', dtype=str, keep_default_na=False)
        part = experimento.carregar_particao_video()
    except (OSError, FileNotFoundError) as e:
        return {'erro': f'corpus ou partição indisponível: {e}'}

    particao_de = part.groupby('video_id')['particao'].agg(lambda s: sorted(set(s))).to_dict()
    ids_corpus = set(corpus['ID'])
    titulos = corpus.drop_duplicates('ID')[['ID', 'TITULO']].reset_index(drop=True)
    tit_norm = [_norm(t) for t in titulos['TITULO']]
    fra_norm = [_norm(t) for t in corpus['FRASE']]

    por_video = []
    for v in videos:
        vid = v['id']
        formato = 'youtube_11' if re.fullmatch(r'[A-Za-z0-9_-]{11}', vid) else 'nao_youtube'
        s_tit, i_tit = _melhor_correspondencia(_norm(v['titulo']), tit_norm)
        melhor_frase = (0.0, None, None)
        for f in v.get('frases_pontuadas', []):
            s, i = _melhor_correspondencia(_norm(f['texto']), fra_norm)
            if i is not None and s > melhor_frase[0]:
                melhor_frase = (s, i, f['texto'])
        vid_tit = titulos['ID'][i_tit] if i_tit is not None else None
        vid_fra = corpus['ID'][melhor_frase[1]] if melhor_frase[1] is not None else None

        if vid in ids_corpus:
            status, vid_corpus = 'confirmado_por_id', vid
        elif (s_tit >= SIM_TITULO and melhor_frase[0] >= SIM_FRASE and vid_tit == vid_fra):
            status, vid_corpus = 'provavel_por_titulo_e_frase', vid_tit
        elif s_tit >= SIM_TITULO or melhor_frase[0] >= SIM_FRASE:
            status = 'possivel_por_titulo_ou_frase'
            vid_corpus = vid_tit if s_tit >= SIM_TITULO else vid_fra
        else:
            status, vid_corpus = 'nao_encontrado', None
        particoes = particao_de.get(vid_corpus, []) if vid_corpus else []
        por_video.append({
            'id_estudo_caso': vid, 'formato_id': formato, 'estrato': v['estrato'],
            'termo': v['termo'], 'posicao_youtube': int(v['posicao_youtube']),
            'titulo': v['titulo'], 'status': status, 'video_id_corpus': vid_corpus,
            'particao': particoes,
            'similaridade_titulo': round(s_tit, 3), 'titulo_corpus_mais_proximo':
                (titulos['TITULO'][i_tit] if i_tit is not None else None),
            'similaridade_frase': round(melhor_frase[0], 3),
            'frase_estudo_caso': melhor_frase[2],
            'frase_corpus_mais_proxima': (corpus['FRASE'][melhor_frase[1]]
                                          if melhor_frase[1] is not None else None),
            'titulo_no_corpus': (titulos.loc[titulos['ID'] == vid_corpus, 'TITULO'].iloc[0]
                                 if vid_corpus in ids_corpus else None),
            'n_frases_no_corpus': int((corpus['ID'] == vid_corpus).sum()) if vid_corpus else 0,
        })

    sobrepostos = [p for p in por_video if p['status'] in STATUS_SOBREPOSTO]
    return {
        'arquivo_corpus': 'data/corpus_toxicidade.csv',
        'arquivo_particao': str(experimento.ARQ_PARTICAO_VIDEO.relative_to(experimento.SRC_ROOT)),
        'criterio': (f"'confirmado_por_id': VIDEO_ID igual ao ID do corpus; "
                     f"'provavel_por_titulo_e_frase': título com similaridade >= {SIM_TITULO} e "
                     f"alguma frase amostrada com similaridade >= {SIM_FRASE}, ambos no mesmo "
                     f"vídeo do corpus; 'possivel_por_titulo_ou_frase': só um dos dois "
                     f"(conferir à mão); os três contam como sobreposição nos subconjuntos, por cautela; similaridade = difflib.ratio sobre "
                     f"texto em minúsculas, sem acentos e sem pontuação"),
        'n_ids_youtube': sum(1 for p in por_video if p['formato_id'] == 'youtube_11'),
        'n_ids_nao_youtube': sum(1 for p in por_video if p['formato_id'] == 'nao_youtube'),
        'nota_ids': ('VIDEO_ID com 16 caracteres não é ID do YouTube (parece o parâmetro si= '
                     'de um link de compartilhamento); para esses vídeos a sobreposição só pode '
                     'ser inferida por título e frase, e um vídeo com título e frases diferentes '
                     'dos do corpus pode, ainda assim, estar nele.'),
        'n_confirmados_por_id': sum(1 for p in por_video if p['status'] == 'confirmado_por_id'),
        'n_provaveis': sum(1 for p in por_video if p['status'] == 'provavel_por_titulo_e_frase'),
        'n_possiveis': sum(1 for p in por_video if p['status'] == 'possivel_por_titulo_ou_frase'),
        'sobrepostos': [{k: p[k] for k in ('id_estudo_caso', 'estrato', 'termo', 'posicao_youtube',
                                           'titulo', 'status', 'video_id_corpus', 'titulo_no_corpus',
                                           'particao', 'similaridade_titulo', 'similaridade_frase',
                                           'n_frases_no_corpus')} for p in sobrepostos],
        'por_video': por_video,
    }


def subconjuntos(videos, sobreposicao):
    """Conjuntos de vídeos para reportar as métricas com e sem sobreposição (M1b)."""
    por_id = {(p['termo'], p['posicao_youtube']): p for p in sobreposicao.get('por_video', [])}

    def status(v):
        return por_id.get((v['termo'], int(v['posicao_youtube'])), {})

    def sobreposto(v):
        return status(v).get('status') in STATUS_SOBREPOSTO

    def no_treino(v):
        return sobreposto(v) and 'treino' in status(v).get('particao', [])

    return {
        'todos': {'descricao': 'os 40 vídeos', 'videos': videos},
        'sem_treino': {'descricao': 'sem os vídeos cujas frases estão na partição de treino do '
                                    'corpus (por ID, ou por título e/ou frase)',
                       'videos': [v for v in videos if not no_treino(v)]},
        'sem_sobreposicao': {'descricao': 'sem nenhum vídeo que esteja no corpus, em qualquer '
                                          'partição (por ID, ou por título e/ou frase)',
                             'videos': [v for v in videos if not sobreposto(v)]},
    }


# =============================================================================
# M4d: perdas no limiar designado
# =============================================================================

CATEGORIAS_PERDA = {
    'sem_linguagem_toxica_nas_frases_amostradas':
        'as frases amostradas não contêm linguagem tóxica',
    'erro_do_classificador':
        'há linguagem tóxica numa frase amostrada e o classificador não a reconheceu',
    'diluicao_na_agregacao':
        'o classificador reconheceu uma frase tóxica, mas a média das três ficou acima do limiar',
    'fora_do_escopo':
        'o problema do vídeo não é toxicidade verbal (ex.: conteúdo sexual)',
}


def classificar_perda(v, julgamento=None):
    """
    Categoria de um vídeo do estrato B não rebaixado no limiar designado.

    Do classificador sai só uma categoria PROVISÓRIA: se nenhuma frase amostrada
    foi predita tóxica, 'sem_linguagem_toxica_nas_frases_amostradas' -- desde que
    o classificador esteja certo; um falso negativo ('erro_do_classificador') só se
    exclui lendo as frases. Se alguma foi predita tóxica, 'diluicao_na_agregacao'.
    A categoria FINAL e o campo 'escopo' vêm do julgamento humano (rótulos da
    anotação cega das frases, --rotulos-humanos, e/ou --julgamentos, chave = ID do
    vídeo no estudo de caso); sem ele ficam null e 'pendente_autor' = true.
    'fora_do_escopo' não é inferido por script.

    Com rótulos frase a frase ('rotulos_frases', na ordem das frases pontuadas), a
    diluição exige que o classificador tenha sinalizado uma frase que os anotadores
    rotularam tóxica; se ele sinalizou só frases que os anotadores rotularam
    Nenhuma e deixou passar a tóxica, a perda é erro do classificador.
    """
    algum_toxico = any(c != 'Nenhuma' for c in v['classes_frases'])
    provisoria = ('diluicao_na_agregacao' if algum_toxico
                  else 'sem_linguagem_toxica_nas_frases_amostradas')
    j = julgamento or {}
    escopo = j.get('escopo')
    tem_tox = j.get('linguagem_toxica_nas_frases')
    rotulos = j.get('rotulos_frases')
    if escopo:
        final = 'fora_do_escopo'
    elif tem_tox is False:
        final = 'sem_linguagem_toxica_nas_frases_amostradas'
    elif tem_tox is True:
        if rotulos is not None:
            reconheceu = any(h != 'Nenhuma' and c != 'Nenhuma'
                             for h, c in zip(rotulos, v['classes_frases']))
        else:
            reconheceu = algum_toxico
        final = 'diluicao_na_agregacao' if reconheceu else 'erro_do_classificador'
    else:
        final = None
    return {
        'categoria_provisoria': provisoria,
        'base_da_provisoria': ('alguma frase amostrada predita Leve/Severa' if algum_toxico
                               else 'as três frases amostradas preditas Nenhuma (vale só se o '
                                    'classificador estiver certo)'),
        'categoria': final, 'escopo': escopo,
        'linguagem_toxica_nas_frases': tem_tox,
        'pendente_autor': final is None,
    }


def detalhe_video(v, tau=None, julgamento=None):
    d = _resumo_video(v)
    d['frases'] = [{'parte': f['parte'], 'texto': f['texto'],
                    'classe_predita': f['classe'], 'score': f['score'],
                    'probs': {'Nenhuma': f['probs'][0], 'Leve': f['probs'][1],
                              'Severa': f['probs'][2]},
                    **({'rotulo_humano': f['humano'], 'anotadores': f['anotadores']}
                       if 'humano' in f else {})}
                   for f in v['frases_pontuadas']]
    if tau is not None:
        d.update(classificar_perda(v, julgamento))
    return d


# =============================================================================
# M4a/M4d/M4f: rótulos humanos das frases amostradas (anotação cega)
# =============================================================================

NIVEIS = ('Nenhuma', 'Leve', 'Severa')
ANOTADORES = ('P1', 'A1', 'P2')
PARTE_DA_JANELA = {'inicio': 'start', 'meio': 'middle', 'fim': 'end'}
ROTULOS_HUMANOS_PADRAO = 'data/reordenacao_rotulos_humanos.csv'
NOTA_ROTULOS_HUMANOS = (
    'As 120 frases amostradas foram rotuladas às cegas pelos três anotadores do corpus (só a '
    'frase, em ordem embaralhada; sem vídeo, busca, estrato ou predição), com o guia do corpus. '
    'Rótulo final: moda das três; sem maioria, o do P1. O rótulo vale para a frase amostrada, '
    'não para o vídeo: não diz nada sobre as partes não amostradas.')


def carregar_rotulos_humanos(caminho, videos):
    """
    Anota cada frase pontuada com o rótulo humano final ('humano') e os três
    rótulos ('anotadores'), a partir do CSV de integrar_rotulos.py. Exige que toda
    frase amostrada tenha rótulo e que o texto coincida com o da planilha de buscas.
    """
    import csv
    with open(caminho, encoding='utf-8-sig') as fh:
        linhas = list(csv.DictReader(fh, delimiter=';'))
    mapa = {(r['video_id'], PARTE_DA_JANELA[r['janela']]): r for r in linhas}
    for v in videos:
        for f in v['frases_pontuadas']:
            r = mapa.get((v['id'], f['parte']))
            if r is None:
                raise SystemExit(f'{caminho}: sem rótulo para {v["id"]} ({f["parte"]})')
            if _norm(r['frase']) != _norm(f['texto']):
                raise SystemExit(f'{caminho}: frase diferente em {v["id"]} ({f["parte"]}): '
                                 f'{r["frase"]!r} x {f["texto"]!r}')
            if r['tox'] not in NIVEIS or any(r[a] not in NIVEIS for a in ANOTADORES):
                raise SystemExit(f'{caminho}: rótulo inválido no item {r.get("item")}')
            f['humano'] = r['tox']
            f['anotadores'] = {a: r[a] for a in ANOTADORES}
    return len(linhas)


def julgamentos_de_rotulos(videos):
    """Julgamento por vídeo derivado dos rótulos das frases (formato de classificar_perda)."""
    return {v['id']: {'linguagem_toxica_nas_frases':
                      any(f['humano'] != 'Nenhuma' for f in v['frases_pontuadas']),
                      'rotulos_frases': [f['humano'] for f in v['frases_pontuadas']]}
            for v in videos if all('humano' in f for f in v['frases_pontuadas'])}


def _kappa_fleiss(tabela):
    """Kappa de Fleiss; tabela = itens x categorias com a contagem de anotadores."""
    tabela = np.asarray(tabela, dtype=float)
    n = tabela.sum(axis=1)[0]
    p_j = tabela.sum(axis=0) / tabela.sum()
    p_i = ((tabela ** 2).sum(axis=1) - n) / (n * (n - 1))
    pe = (p_j ** 2).sum()
    return float((p_i.mean() - pe) / (1 - pe))


def _kappa_cohen(x, y):
    po = float(np.mean([a == b for a, b in zip(x, y)]))
    pe = sum(np.mean([a == k for a in x]) * np.mean([b == k for b in y]) for k in NIVEIS)
    return float((po - pe) / (1 - pe)), po


def _metricas_por_classe(y, p, classes):
    """Precisão, revocação e F1 por classe (zero quando indefinido) e o macro F1."""
    por = {}
    for c in classes:
        tp = sum(a == c and b == c for a, b in zip(y, p))
        fp = sum(a != c and b == c for a, b in zip(y, p))
        fn = sum(a == c and b != c for a, b in zip(y, p))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        por[c] = {'precisao': pr, 'revocacao': rc,
                  'f1': 2 * pr * rc / (pr + rc) if pr + rc else 0.0, 'suporte': tp + fn}
    return por, float(np.mean([por[c]['f1'] for c in classes]))


def _confusao(y, p):
    return {h: {c: sum(a == h and b == c for a, b in zip(y, p)) for c in NIVEIS} for h in NIVEIS}


def avaliar_rotulos_humanos(videos, A, B, tau, rng, limiares=()):
    """
    Humano x modelo nas frases amostradas, por estrato e no total, e o que os
    rótulos dizem dos vídeos: quantos têm alguma frase tóxica, a AUC do score do
    vídeo contra esse rótulo, e se as frases que causaram os falsos alarmes do
    estrato A são tóxicas para os anotadores (M4f).
    """
    frases = [(v, f) for v in videos for f in v['frases_pontuadas']]
    y = [f['humano'] for _, f in frases]
    p = [f['classe'] for _, f in frases]
    tab = [[sum(f['anotadores'][a] == k for a in ANOTADORES) for k in NIVEIS] for _, f in frases]
    maximo = [max(t) for t in tab]
    pares = {}
    for i, a in enumerate(ANOTADORES):
        for b in ANOTADORES[i + 1:]:
            k, po = _kappa_cohen([f['anotadores'][a] for _, f in frases],
                                 [f['anotadores'][b] for _, f in frases])
            pares[f'{a}x{b}'] = {'kappa_cohen': k, 'concordancia': po}

    por_classe, macro = _metricas_por_classe(y, p, NIVEIS)
    yb = ['Toxica' if h != 'Nenhuma' else 'Nenhuma' for h in y]
    pb = ['Toxica' if c != 'Nenhuma' else 'Nenhuma' for c in p]
    por_bin, macro_bin = _metricas_por_classe(yb, pb, ('Nenhuma', 'Toxica'))
    res = {
        'nota': NOTA_ROTULOS_HUMANOS,
        'n_frases': len(frases), 'n_videos': len(videos),
        'concordancia': {'kappa_fleiss': _kappa_fleiss(tab),
                         'unanimes': sum(m == 3 for m in maximo),
                         'sem_maioria': sum(m == 1 for m in maximo),
                         'pares': pares},
        'total': {'acuracia': float(np.mean([a == b for a, b in zip(y, p)])),
                  'macro_f1': macro, 'por_classe': por_classe,
                  'binario': {'macro_f1': macro_bin, 'por_classe': por_bin},
                  'confusao_humano_x_predito': _confusao(y, p)},
        'por_estrato': {},
    }
    for nome, grupo in (('A_responsavel', A), ('B_apelo_infantil', B)):
        fg = [f for v in grupo for f in v['frases_pontuadas']]
        yg, pg = [f['humano'] for f in fg], [f['classe'] for f in fg]
        tox_h = [f for f in fg if f['humano'] != 'Nenhuma']
        nen_h = [f for f in fg if f['humano'] == 'Nenhuma']
        res['por_estrato'][nome] = {
            'n_frases': len(fg),
            'distribuicao_humano': {k: yg.count(k) for k in NIVEIS},
            'distribuicao_predita': {k: pg.count(k) for k in NIVEIS},
            'acuracia': float(np.mean([a == b for a, b in zip(yg, pg)])),
            'confusao_humano_x_predito': _confusao(yg, pg),
            'toxicas_humano_sinalizadas': sum(f['classe'] != 'Nenhuma' for f in tox_h),
            'toxicas_humano': len(tox_h),
            'nenhuma_humano_sinalizadas': sum(f['classe'] != 'Nenhuma' for f in nen_h),
            'nenhuma_humano': len(nen_h),
            'videos': len(grupo),
            'videos_com_frase_toxica_humano': sum(
                any(f['humano'] != 'Nenhuma' for f in v['frases_pontuadas']) for v in grupo),
            'videos_com_frase_sinalizada': sum(
                any(f['classe'] != 'Nenhuma' for f in v['frases_pontuadas']) for v in grupo),
        }

    # Vídeo "com linguagem tóxica nas frases amostradas" pelo rótulo humano x score.
    pos = [v['score_media'] for v in videos
           if any(f['humano'] != 'Nenhuma' for f in v['frases_pontuadas'])]
    neg = [v['score_media'] for v in videos
           if not any(f['humano'] != 'Nenhuma' for f in v['frases_pontuadas'])]
    auc, lo, hi = auc_com_ic(neg, pos, rng)
    res['auc_rotulo_humano_do_video'] = {
        'positivos': len(pos), 'negativos': len(neg), 'auc': auc, 'auc_ic95': [lo, hi],
        'nota': ('positivo = vídeo com alguma frase amostrada rotulada tóxica pelos anotadores; '
                 'mede o score contra os rótulos das MESMAS frases que ele agrega, não a '
                 'toxicidade do vídeo inteiro')}

    # Falsos alarmes do estrato A: as frases que o classificador sinalizou são tóxicas
    # para os anotadores? "Erro do classificador" = todas as sinalizadas são Nenhuma.
    def falsos_alarme(t):
        fa = []
        for v in sorted([v for v in A if v['score_media'] < t], key=lambda v: v['score_media']):
            sinal = [{'parte': f['parte'], 'texto': f['texto'], 'classe_predita': f['classe'],
                      'rotulo_humano': f['humano']}
                     for f in v['frases_pontuadas'] if f['classe'] != 'Nenhuma']
            fa.append({'id': v['id'], 'titulo': v['titulo'], 'score': v['score_media'],
                       'frases_sinalizadas': sinal,
                       'erro_do_classificador': bool(sinal) and all(
                           s['rotulo_humano'] == 'Nenhuma' for s in sinal)})
        return {'limiar': float(t), 'n': len(fa),
                'por_erro_do_classificador': sum(x['erro_do_classificador'] for x in fa),
                'sem_frase_sinalizada': sum(not x['frases_sinalizadas'] for x in fa),
                'detalhe': fa}

    res['falsos_alarme_a'] = falsos_alarme(tau)
    res['falsos_alarme_a_por_limiar'] = [falsos_alarme(t) for t in limiares]
    # Frases do estrato A que os anotadores rotularam tóxicas (resultado infantil com
    # linguagem imprópria, M4a).
    res['estrato_a_frases_toxicas_humano'] = [
        {'id': v['id'], 'titulo': v['titulo'], 'termo': v['termo'], 'parte': f['parte'],
         'texto': f['texto'], 'rotulo_humano': f['humano'], 'anotadores': f['anotadores'],
         'classe_predita': f['classe'], 'score_video': v['score_media'],
         'rebaixado': v['score_media'] < tau}
        for v in A for f in v['frases_pontuadas'] if f['humano'] != 'Nenhuma']
    return res


# =============================================================================
# Reordenação: deslocamentos, banda e sensibilidade
# =============================================================================

def reordenar(buscas, score_de, delta=None):
    """
    Posição na plataforma (de) e após a reordenação (para) de cada vídeo com
    score. Sem delta: ordenação estável por score decrescente; com delta: banda de
    tolerância (analisar_reordenacao.ordenar_com_banda).
    """
    reord = importar_irmao('analisar_reordenacao')
    entradas = []
    for b in buscas:
        com = sorted([v for v in b['videos'] if score_de(v) is not None],
                     key=lambda v: int(v['posicao_youtube']))
        rank_yt = {id(v): i for i, v in enumerate(com, 1)}
        if delta is None:
            ordem = sorted(com, key=lambda v: -score_de(v))
        else:
            ordem = reord.ordenar_com_banda(com, score_de, lambda v: rank_yt[id(v)], delta)
        for novo, v in enumerate(ordem, 1):
            de = rank_yt[id(v)]
            entradas.append({'termo': b['termo'], 'estrato': b['estrato'], 'id': v['id'],
                             'titulo': v['titulo'], 'de': de, 'para': novo,
                             'delta': novo - de, 'score': score_de(v)})
    return entradas


def resumo_reordenacao(entradas, tau=None):
    """Deslocamento por estrato e, no B, o que acontece com os vídeos rebaixados."""
    from scipy.stats import spearmanr

    out = {}
    for estrato in dict.fromkeys(e['estrato'] for e in entradas):
        es = [e for e in entradas if e['estrato'] == estrato]
        termos = list(dict.fromkeys(e['termo'] for e in es))
        rhos, top3, primeiro = [], [], 0
        for t in termos:
            et = sorted([e for e in es if e['termo'] == t], key=lambda e: e['de'])
            if len(et) >= 3:
                rhos.append(float(spearmanr([e['de'] for e in et], [e['para'] for e in et])[0]))
                topo_yt = {e['id'] for e in et if e['de'] <= 3}
                topo_novo = {e['id'] for e in et if e['para'] <= 3}
                top3.append(len(topo_yt & topo_novo) / 3)
            primeiro += int(not any(e['de'] == 1 and e['para'] == 1 for e in et))
        r = {'n_videos': len(es),
             'deslocamento_medio_abs': float(np.mean([abs(e['delta']) for e in es])),
             'deslocamento_maximo_abs': int(max(abs(e['delta']) for e in es)),
             'n_posicoes_alteradas': sum(1 for e in es if e['delta'] != 0),
             'n_desceram': sum(1 for e in es if e['delta'] > 0),
             'spearman_medio': float(np.mean(rhos)) if rhos else None,
             'sobreposicao_top3_media': float(np.mean(top3)) if top3 else None,
             'buscas_com_novo_primeiro': primeiro, 'n_buscas': len(termos)}
        if tau is not None:
            reb = [e for e in es if e['score'] < tau]
            r['rebaixados_no_limiar'] = {
                'limiar': tau, 'n': len(reb),
                'n_desceram_na_lista': sum(1 for e in reb if e['delta'] > 0),
                'deslocamento_medio_assinado': (float(np.mean([e['delta'] for e in reb]))
                                                if reb else None),
                'posicao_media_antes': float(np.mean([e['de'] for e in reb])) if reb else None,
                'posicao_media_depois': float(np.mean([e['para'] for e in reb])) if reb else None,
            }
        out[estrato] = r
    return out


def sensibilidade_pesos_agregacao(buscas, A, B):
    """AUC e deslocamento médio para cada peso de Leve e regra de agregação (M7b)."""
    regras = importar_irmao('simular_amostragem').REGRAS
    rng = np.random.default_rng([SEED, 7])
    linhas = []
    for w in PESOS_LEVE_SENS:
        for regra, f_r in regras.items():
            def s_de(v, w=w, f_r=f_r):
                sc = [score_frase(f['probs'], w) for f in v['frases_pontuadas']]
                return float(f_r(np.array([sc]))[0]) if sc else None
            sa = [s_de(v) for v in A]
            sb = [s_de(v) for v in B]
            auc, lo, hi = auc_com_ic(sa, sb, rng, N_BOOT_SENS)
            rr = resumo_reordenacao(reordenar(buscas, s_de))
            linhas.append({
                'peso_leve': w, 'agregacao': regra, 'auc': auc, 'auc_ic95': [lo, hi],
                'media_a': float(np.mean(sa)), 'media_b': float(np.mean(sb)),
                'deslocamento_medio_abs_A': rr.get('A_responsavel', {}).get('deslocamento_medio_abs'),
                'deslocamento_medio_abs_B': rr.get('B_apelo_infantil', {}).get('deslocamento_medio_abs'),
                'referencia': (w == PESOS['Leve'] and regra == 'media'),
            })
    return {'n_bootstrap_auc': N_BOOT_SENS, 'pesos_leve': list(PESOS_LEVE_SENS),
            'agregacoes': list(regras), 'linhas': linhas,
            'nota': 'Deslocamento: média de |posição na plataforma - posição pelo score| no '
                    'estrato, ordenação estável por score, sem banda.'}


# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Separação entre estratos de busca')
    parser.add_argument('--csv', default='data/reordenacao_buscas.csv')
    parser.add_argument('--idioma', choices=['pt', 'en'], default='pt',
                        help='Idioma dos rótulos das figuras e do texto das tabelas')
    parser.add_argument('--modelo', default=None,
                        help='Pasta do modelo (padrão: o de referência da rodada)')
    parser.add_argument('--predicoes', default=None,
                        help='estratos_*.json com as probabilidades por frase; dispensa o modelo')
    parser.add_argument('--resultados', default=None,
                        help='Pasta onde procurar a simulação e as OOF (padrão: results/<tag>)')
    parser.add_argument('--simulacao', default=None,
                        help='simulacao_amostragem_*.json explícito (padrão: o mais recente)')
    parser.add_argument('--julgamentos', default=None,
                        help='JSON {id_do_video: {"linguagem_toxica_nas_frases": bool|null, '
                             '"escopo": str|null}} com a leitura humana das perdas (M4d)')
    parser.add_argument('--rotulos-humanos', default=ROTULOS_HUMANOS_PADRAO,
                        help='CSV da anotação cega das frases amostradas (integrar_rotulos.py); '
                             'ignorado se não existir; "" desliga')
    parser.add_argument('--saida', default=None,
                        help='Pasta de saída do JSON; tabelas e figuras em <saida>/latex/ '
                             '(padrão: results/<tag> e experimento.dir_latex)')
    args = parser.parse_args()

    planilha = importar_irmao('analisar_reordenacao').carregar_planilha
    saidas = importar_irmao('regerar_tabelas_estudo_caso')
    pasta_res = Path(args.resultados) if args.resultados else experimento.dir_resultados()
    pasta_saida = Path(args.saida) if args.saida else experimento.dir_resultados()
    pasta_saida.mkdir(parents=True, exist_ok=True)

    buscas = planilha(args.csv)
    if args.predicoes:
        with open(args.predicoes, encoding='utf-8') as fh:
            predicoes = json.load(fh)
        nome_modelo = predicoes.get('modelo') or f'predições de {Path(args.predicoes).name}'
        pontuar_videos(buscas, predicoes=predicoes)
        print(f'Predições reaproveitadas de {args.predicoes}\n')
    else:
        modelo, caminho = carregar_modelo(args.modelo)
        nome_modelo = caminho.name
        pontuar_videos(buscas, modelo=modelo)

    julgamentos = {}
    if args.julgamentos:
        with open(args.julgamentos, encoding='utf-8') as fh:
            julgamentos = json.load(fh)

    videos = [v for b in buscas for v in b['videos'] if v['score_media'] is not None]
    sem_score = [v for b in buscas for v in b['videos'] if v['score_media'] is None]
    A = [v for v in videos if v['estrato'] == 'A_responsavel']
    B = [v for v in videos if v['estrato'] == 'B_apelo_infantil']

    # Rótulos humanos das frases: fecham as categorias das perdas; um julgamento
    # explícito (--julgamentos, ex. 'escopo') prevalece sobre o derivado.
    com_rotulos = bool(args.rotulos_humanos) and Path(args.rotulos_humanos).exists()
    if com_rotulos:
        n_rot = carregar_rotulos_humanos(args.rotulos_humanos, videos)
        derivados = julgamentos_de_rotulos(videos)
        for vid, j in julgamentos.items():
            derivados.setdefault(vid, {}).update(j)
        julgamentos = derivados
        print(f'Rótulos humanos: {args.rotulos_humanos} ({n_rot} frases)\n')
    elif args.rotulos_humanos:
        print(f'Rótulos humanos: {args.rotulos_humanos} não encontrado; categorias provisórias\n')

    limiar, banda = obter_limiar_e_banda(pasta_res, args.simulacao)
    tau = limiar['valor']
    a_priori = limiar['origem'] != 'legado'
    grade = sorted(set(LIMIARES_GRADE) | {tau}, reverse=True)

    resultado = {'timestamp': datetime.now().isoformat(), 'origem': args.csv,
                 'modelo': nome_modelo, 'n_buscas': len(buscas), 'n_videos': len(videos),
                 'n_videos_sem_transcricao': len(sem_score),
                 'score': ('filtro de toxicidade isolado: média de 1 - (0,5*p_leve + '
                           '1,0*p_severa) nas frases amostradas'),
                 'regra_de_decisao': 'score < limiar => rebaixado',
                 'nota_estratos': ('Estratos definidos pela busca, não por anotação dos vídeos: '
                                   'a AUC mede a separação entre estratos de busca (M4a).'),
                 'nota_partes_nao_amostradas': NOTA_PARTES_NAO_AMOSTRADAS}

    # ---- distribuição por estrato ----
    print('=' * 78)
    print('DISTRIBUIÇÃO DO SCORE POR ESTRATO (só filtro de toxicidade)')
    print('=' * 78)
    print(f'{"estrato / termo":<34} {"n":>3} {"média":>7} {"mín":>7} {"máx":>7} {"amplitude":>10}')
    print('-' * 78)
    por_termo = {}
    for b in buscas:
        s = [v['score_media'] for v in b['videos'] if v['score_media'] is not None]
        por_termo[b['termo']] = {'estrato': b['estrato'], 'n': len(s),
                                 'media': float(np.mean(s)), 'min': float(min(s)),
                                 'max': float(max(s)), 'amplitude': float(max(s) - min(s))}
        print(f'  {b["termo"][:32]:<32} {len(s):>3} {np.mean(s):>7.3f} '
              f'{min(s):>7.3f} {max(s):>7.3f} {max(s)-min(s):>10.3f}')
    print('-' * 78)
    for nome, grupo in (('A_responsavel', A), ('B_apelo_infantil', B)):
        s = [v['score_media'] for v in grupo]
        resultado[f'estrato_{nome}'] = {'n': len(s), 'media': float(np.mean(s)),
                                        'min': float(min(s)), 'max': float(max(s)),
                                        'amplitude': float(max(s) - min(s))}
        print(f'{nome:<34} {len(s):>3} {np.mean(s):>7.3f} {min(s):>7.3f} '
              f'{max(s):>7.3f} {max(s)-min(s):>10.3f}')
    resultado['por_termo'] = por_termo

    # ---- separação entre estratos (mesma sequência de sorteios da versão submetida) ----
    rng = np.random.default_rng(SEED)
    sep_media = separacao(A, B, rng, grade, 'score_media')
    sep_min = separacao(A, B, rng, grade, 'score_minimo')
    resultado['separacao_media'] = {k: sep_media[k] for k in
                                    ('mannwhitney_u', 'p', 'auc', 'auc_ic95', 'diferenca_medias')}
    resultado['separacao_minimo'] = {k: sep_min[k] for k in
                                     ('mannwhitney_u', 'p', 'auc', 'auc_ic95')}
    print('\n' + '=' * 78)
    print('SEPARAÇÃO ENTRE ESTRATOS (A deveria pontuar acima de B)')
    print('=' * 78)
    print(f"  diferença de médias:  {sep_media['diferenca_medias']:+.4f}")
    print(f"  Mann-Whitney:         U = {sep_media['mannwhitney_u']:.1f}, p = {sep_media['p']:.5f}")
    print(f"  AUC (média das 3):    {sep_media['auc']:.3f}  IC95% "
          f"[{sep_media['auc_ic95'][0]:.3f}, {sep_media['auc_ic95'][1]:.3f}]")
    print(f"  AUC (mínimo das 3):   {sep_min['auc']:.3f}  IC95% "
          f"[{sep_min['auc_ic95'][0]:.3f}, {sep_min['auc_ic95'][1]:.3f}]  (p = {sep_min['p']:.5f})")

    # ---- limiares (M4b, M4c) ----
    deteccao = []
    for e in sep_media['por_limiar']:
        e = dict(e)
        e['a_priori'] = bool(a_priori and e['limiar'] == tau)
        e['exploratorio'] = not e['a_priori']
        deteccao.append(e)
    resultado['deteccao_por_limiar'] = deteccao
    designado = next(e for e in deteccao if e['limiar'] == tau)
    resultado['limiar_designado'] = dict(limiar, resultado={
        k: designado[k] for k in ('detectados_b', 'total_b', 'falsos_a', 'total_a')})

    print('\n' + '=' * 78)
    print(f"DETECÇÃO POR LIMIAR (regra: score < limiar => rebaixado; designado = "
          f"{tau:.4f}, origem {limiar['origem']})")
    print('=' * 78)
    print(f'{"limiar":>8} {"":>12} {"rebaixados em B":>17} {"falso alarme em A":>19}')
    for e in deteccao:
        marca = 'A PRIORI' if e['a_priori'] else 'exploratório'
        print(f"{e['limiar']:>8.4f} {marca:>12} "
              f"{e['detectados_b']:>11}/{e['total_b']:<5} {e['falsos_a']:>13}/{e['total_a']}")

    # ---- perdas no limiar designado (M4d) ----
    perdas = sorted([v for v in B if v['score_media'] >= tau], key=lambda v: v['score_media'])
    det_perdas = [detalhe_video(v, tau, julgamentos.get(v['id'])) for v in perdas]
    contagem = {}
    for d in det_perdas:
        contagem[d['categoria_provisoria']] = contagem.get(d['categoria_provisoria'], 0) + 1
    resultado['falhas_estrato_b'] = {
        'limiar': tau, 'limiar_a_priori': a_priori,
        'nao_detectados': len(perdas), 'total': len(B),
        'por_categoria_provisoria': contagem,
        'por_categoria_final': {c: sum(1 for d in det_perdas if d['categoria'] == c)
                                for c in CATEGORIAS_PERDA},
        'pendentes_autor': sum(1 for d in det_perdas if d['pendente_autor']),
        'categorias': CATEGORIAS_PERDA,
        'nota': NOTA_PARTES_NAO_AMOSTRADAS + ' A categoria final e o escopo dependem da leitura '
                'humana das frases (--rotulos-humanos, --julgamentos); fora_do_escopo nunca é '
                'inferido por script.',
        'fonte_categoria_final': (args.rotulos_humanos if com_rotulos
                                  else (args.julgamentos or None)),
        'detalhe': det_perdas,
    }
    print('\n' + '=' * 78)
    print(f'PERDAS NO ESTRATO B (limiar designado {tau:.4f})')
    print('=' * 78)
    print(f'  não rebaixados: {len(perdas)} de {len(B)}  | provisório: {contagem}')
    if com_rotulos:
        print(f"  final (rótulos humanos): "
              f"{ {c: n for c, n in resultado['falhas_estrato_b']['por_categoria_final'].items() if n} }")
    for d in det_perdas:
        print(f"\n    [{d['categoria'] or d['categoria_provisoria']}] {d['score']:.3f}  "
              f"{d['titulo'][:60]}")
        for f in d['frases']:
            hum = f" humano {f['rotulo_humano'][:7]:<7}" if 'rotulo_humano' in f else ''
            print(f"       {f['score']:.3f} ({f['classe_predita'][:7]:<7}){hum} {f['texto'][:60]}")

    # ---- falsos alarmes no estrato A ----
    fa = sorted([v for v in A if v['score_media'] < tau], key=lambda v: v['score_media'])
    resultado['falsos_alarme_a'] = [detalhe_video(v) for v in fa]
    print(f'\n  falsos alarmes no estrato A (limiar {tau:.4f}): {len(fa)} de {len(A)}')
    for d in resultado['falsos_alarme_a']:
        print(f"    {d['score']:.3f}  {d['titulo'][:60]}")
        for f in d['frases']:
            hum = f" humano {f['rotulo_humano'][:7]:<7}" if 'rotulo_humano' in f else ''
            print(f"       {f['score']:.3f} ({f['classe_predita'][:7]:<7}){hum} {f['texto'][:60]}")

    # ---- rótulos humanos das frases amostradas (M4a, M4d, M4f) ----
    resultado['rotulos_humanos'] = None
    if com_rotulos:
        rh = avaliar_rotulos_humanos(videos, A, B, tau, np.random.default_rng([SEED, 200]), grade)
        resultado['rotulos_humanos'] = dict(rh, origem=args.rotulos_humanos)
        c = rh['concordancia']
        print('\n' + '=' * 78)
        print('RÓTULOS HUMANOS DAS FRASES AMOSTRADAS (anotação cega)')
        print('=' * 78)
        print(f"  {rh['n_frases']} frases | kappa de Fleiss {c['kappa_fleiss']:.3f} | unânimes "
              f"{c['unanimes']} | sem maioria {c['sem_maioria']}")
        t = rh['total']
        print(f"  humano x modelo: acurácia {t['acuracia']:.3f}, macro F1 {t['macro_f1']:.3f}, "
              f"binário {t['binario']['macro_f1']:.3f}")
        for nome, e in rh['por_estrato'].items():
            print(f"  {nome:<17} humano {e['distribuicao_humano']} | predito "
                  f"{e['distribuicao_predita']} | tóxicas sinalizadas "
                  f"{e['toxicas_humano_sinalizadas']}/{e['toxicas_humano']} | Nenhuma sinalizadas "
                  f"{e['nenhuma_humano_sinalizadas']}/{e['nenhuma_humano']} | vídeos com frase "
                  f"tóxica {e['videos_com_frase_toxica_humano']}/{e['videos']}")
        ah = rh['auc_rotulo_humano_do_video']
        print(f"  AUC do score x rótulo humano do vídeo: {ah['auc']:.3f} "
              f"[{ah['auc_ic95'][0]:.3f}, {ah['auc_ic95'][1]:.3f}] "
              f"({ah['positivos']} positivos, {ah['negativos']} negativos)")
        for x in [rh['falsos_alarme_a']] + rh['falsos_alarme_a_por_limiar']:
            print(f"  limiar {x['limiar']:.4f}: falsos alarmes de A {x['n']}, por erro do "
                  f"classificador {x['por_erro_do_classificador']}, sem frase sinalizada "
                  f"{x['sem_frase_sinalizada']}")
        for x in rh['estrato_a_frases_toxicas_humano']:
            print(f"  A, frase tóxica p/ anotadores: [{x['rotulo_humano']}; modelo "
                  f"{x['classe_predita']}] {x['texto'][:60]}  ({x['titulo'][:40]})")

    # ---- sobreposição com o corpus e métricas por subconjunto (M1b) ----
    sobre = sobreposicao_corpus(videos)
    resultado['sobreposicao_corpus'] = sobre
    resultado['metricas_por_subconjunto'] = {}
    print('\n' + '=' * 78)
    print('SOBREPOSIÇÃO COM O CORPUS v1 E MÉTRICAS POR SUBCONJUNTO (M1b)')
    print('=' * 78)
    for p in sobre.get('sobrepostos', []):
        print(f"  {p['status']:<28} {p['estrato'][:1]} {p['termo'][:24]:<24} pos {p['posicao_youtube']:>2} "
              f"-> {p['video_id_corpus']} ({', '.join(p['particao']) or '?'})  {p['titulo'][:40]}")
    for i, (nome, sub) in enumerate(subconjuntos(videos, sobre).items()):
        sa = [v for v in sub['videos'] if v['estrato'] == 'A_responsavel']
        sb = [v for v in sub['videos'] if v['estrato'] == 'B_apelo_infantil']
        m = separacao(sa, sb, np.random.default_rng([SEED, 100 + i]), [tau] + list(LIMIARES_GRADE))
        m['descricao'] = sub['descricao']
        no_sub = {id(v) for v in sub['videos']}
        m['ids_excluidos'] = [v['id'] for v in videos if id(v) not in no_sub]
        m['limiar_designado'] = {k: m['por_limiar'][0][k]
                                 for k in ('limiar', 'detectados_b', 'total_b', 'falsos_a', 'total_a')}
        resultado['metricas_por_subconjunto'][nome] = m
        print(f"  {nome:<17} nA={m['n_a']:>2} nB={m['n_b']:>2}  AUC {m['auc']:.3f} "
              f"[{m['auc_ic95'][0]:.3f}, {m['auc_ic95'][1]:.3f}]  U={m['mannwhitney_u']:.1f} "
              f"p={m['p']:.2e}  em {tau:.4f}: B {m['limiar_designado']['detectados_b']}/{m['n_b']}, "
              f"FA A {m['limiar_designado']['falsos_a']}/{m['n_a']}")

    # ---- deslocamentos sem banda (chave legada) e com banda (M7c) ----
    s_media = (lambda v: v['score_media'])
    resultado['deslocamentos'] = reordenar(buscas, s_media)
    sem_banda = resumo_reordenacao(resultado['deslocamentos'], tau)
    print('\n' + '=' * 78)
    print('REORDENAÇÃO: SEM BANDA E COM BANDA DE TOLERÂNCIA (M7c)')
    print('=' * 78)
    if banda.get('delta') is not None:
        delta = float(banda['delta'])
        com = reordenar(buscas, s_media, delta)
        sens = []
        for q, val in (banda.get('quantis_diferenca') or {}).items():
            rr = resumo_reordenacao(reordenar(buscas, s_media, float(val)), tau)
            sens.append({'quantil_dif_entre_videos_apropriados': q, 'delta': float(val),
                         'resumo': rr})
        resultado['banda_tolerancia'] = dict(
            banda, regra=('u fica acima de v se S(u) - S(v) > delta; nos demais pares vale a '
                          'ordem da plataforma (extensão linear lexicograficamente mínima; ver '
                          'analisar_reordenacao.ordenar_com_banda)'),
            sem_banda=sem_banda, com_banda=resumo_reordenacao(com, tau),
            deslocamentos=com, sensibilidade_delta=sens)
        for nome, r in (('sem banda', sem_banda), (f'delta = {delta:.4f}',
                                                   resultado['banda_tolerancia']['com_banda'])):
            a, b = r.get('A_responsavel', {}), r.get('B_apelo_infantil', {})
            rb = b.get('rebaixados_no_limiar', {})
            print(f"  {nome:<16} A: desloc. {a.get('deslocamento_medio_abs', 0):.2f}, "
                  f"{a.get('n_posicoes_alteradas')} posições alteradas | B: desloc. "
                  f"{b.get('deslocamento_medio_abs', 0):.2f}, rebaixados no limiar que desceram "
                  f"{rb.get('n_desceram_na_lista')}/{rb.get('n')}")
    else:
        resultado['banda_tolerancia'] = None
        print('  sem delta (simulação indisponível): banda não avaliada')
    resultado['reordenacao_sem_banda'] = sem_banda

    # ---- sensibilidade a pesos e agregação (M7b) ----
    resultado['sensibilidade_pesos_agregacao'] = sensibilidade_pesos_agregacao(buscas, A, B)
    print('\n' + '=' * 78)
    print('SENSIBILIDADE: PESO DE LEVE x AGREGAÇÃO (M7b)')
    print('=' * 78)
    for ln in resultado['sensibilidade_pesos_agregacao']['linhas']:
        print(f"  w_leve={ln['peso_leve']:<5} {ln['agregacao']:<14} AUC {ln['auc']:.3f} "
              f"[{ln['auc_ic95'][0]:.3f}, {ln['auc_ic95'][1]:.3f}]  desloc. A "
              f"{ln['deslocamento_medio_abs_A']:.2f}  B {ln['deslocamento_medio_abs_B']:.2f}")

    # ---- probabilidades por frase, para reanálise sem modelo (--predicoes) ----
    resultado['videos'] = [{
        'termo': v['termo'], 'estrato': v['estrato'], 'posicao_youtube': int(v['posicao_youtube']),
        'id': v['id'], 'titulo': v['titulo'], 'score_media': v['score_media'],
        'score_minimo': v['score_minimo'],
        'frases_pontuadas': [{'parte': f['parte'], 'texto': f['texto'], 'probs': f['probs'],
                              'classe': f['classe'], 'score': f['score'],
                              **({'humano': f['humano']} if 'humano' in f else {})}
                             for f in v['frases_pontuadas']]} for v in videos]

    destino = pasta_saida / f'estratos_{datetime.now():%Y%m%d_%H%M%S}.json'
    with open(destino, 'w', encoding='utf-8') as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False, default=str)
    print(f'\nSalvo em: {destino}')

    for caminho in saidas.escrever_saidas_estratos(resultado, args.idioma, args.saida):
        print(f'Saída LaTeX: {caminho}')


if __name__ == '__main__':
    main()
