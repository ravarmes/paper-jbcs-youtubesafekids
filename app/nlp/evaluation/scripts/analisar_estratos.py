"""
Estudo de caso: o score separa buscas de risco de buscas seguras?

A análise de reordenação (`analisar_reordenacao.py`) mede o quanto a ordem muda
dentro de cada busca. Esta aqui responde à pergunta anterior a essa: o score
distingue conteúdo que uma criança encontraria e não deveria ver daquele que um
responsável procuraria deliberadamente?

Os dois estratos medem coisas diferentes e não são fundidos numa média:
  A_responsavel   — consultas que um adulto digitaria buscando conteúdo infantil;
                    o desejável é *não* reordenar, porque não há risco a corrigir;
  B_apelo_infantil — animações adultas com apelo visual infantil, que uma criança
                    plausivelmente clicaria; o desejável é rebaixá-las.

Reporta, no nível do vídeo:
  - distribuição do score por estrato (média, amplitude);
  - separação entre estratos: Mann-Whitney e AUC com IC por bootstrap;
  - detecção por limiar, com o falso alarme correspondente no estrato seguro;
  - taxonomia das falhas: distingue vídeo não detectado porque as três frases
    amostradas nada continham (falha de amostragem) de vídeo cujas frases
    continham agressão que o classificador não reconheceu (falha do modelo).

Uso:
    python -m app.nlp.evaluation.scripts.analisar_estratos --csv data/reordenacao_buscas.csv
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PESOS = {'Nenhuma': 0.0, 'Leve': 0.5, 'Severa': 1.0}
N_BOOTSTRAP = 10000
SEED = 42

# Idioma dos rotulos da figura e do texto da tabela (--idioma). Os termos de
# busca nunca sao traduzidos: sao entrada do sistema.
IDIOMA = 'pt'

# Limiar de referência para "sinalizar como suspeito". Escolhido abaixo do piso
# observado no estrato seguro, e reportado junto com o falso alarme que produz.
LIMIARES = (0.90, 0.85, 0.80, 0.70, 0.50)


def carregar_modelo():
    from app.nlp.models.bertimbau_toxicity import BertimbauToxicity
    trained = Path(project_root) / 'app' / 'nlp' / 'models' / 'trained'
    candidatos = [p for p in trained.glob('TOX_*') if (p / 'config.json').exists()]
    if not candidatos:
        raise FileNotFoundError(f'Nenhum modelo TOX_* treinado em {trained}')
    caminho = max(candidatos, key=lambda p: p.stat().st_mtime)
    print(f'Modelo: {caminho.name}\n')
    return BertimbauToxicity(model_path=str(caminho))


def pontuar(modelo, texto):
    """Devolve (score, classe predita) de uma sentença."""
    r = modelo.predict_toxicity(texto, return_probabilities=True)
    probs = r['probabilities']
    return 1.0 - sum(probs[c] * PESOS[c] for c in PESOS), max(probs, key=probs.get)


def auc_com_ic(scores_a, scores_b, rng):
    """
    AUC de separação entre estratos, com IC 95% por bootstrap.

    O estrato B é o "positivo" (deveria ser rebaixado) e o score é invertido,
    de modo que AUC = 1 significa separação perfeita.
    """
    from sklearn.metrics import roc_auc_score

    y = np.array([0] * len(scores_a) + [1] * len(scores_b))
    s = -np.array(list(scores_a) + list(scores_b))
    ponto = float(roc_auc_score(y, s))

    amostras = []
    for _ in range(N_BOOTSTRAP):
        i = rng.integers(0, len(s), len(s))
        if len(np.unique(y[i])) < 2:
            continue
        amostras.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(amostras, [2.5, 97.5])
    return ponto, float(lo), float(hi)


def main():
    from scipy.stats import mannwhitneyu

    from app.nlp.evaluation.scripts.analisar_reordenacao import carregar_planilha

    global IDIOMA

    parser = argparse.ArgumentParser(description='Separação entre estratos de busca')
    parser.add_argument('--csv', default='data/reordenacao_buscas.csv')
    parser.add_argument('--idioma', choices=['pt', 'en'], default='pt',
                        help='Idioma dos rótulos da figura e do texto da tabela')
    args = parser.parse_args()
    IDIOMA = args.idioma

    modelo = carregar_modelo()
    buscas = carregar_planilha(args.csv)
    rng = np.random.default_rng(SEED)

    # ---- pontuação, guardando também as classes preditas de cada frase ----
    for busca in buscas:
        for v in busca['videos']:
            frases = [t for t in v['frases'].values() if t.strip()]
            pares = [pontuar(modelo, t) for t in frases]
            v['scores_frases'] = [p[0] for p in pares]
            v['classes_frases'] = [p[1] for p in pares]
            v['score_media'] = float(np.mean(v['scores_frases'])) if pares else None
            v['score_minimo'] = float(np.min(v['scores_frases'])) if pares else None
            v['tem_frase_toxica'] = any(c != 'Nenhuma' for c in v['classes_frases'])

    videos = [v for b in buscas for v in b['videos']]
    for b in buscas:
        for v in b['videos']:
            v['estrato'], v['termo'] = b['estrato'], b['termo']

    A = [v for v in videos if v['estrato'] == 'A_responsavel']
    B = [v for v in videos if v['estrato'] == 'B_apelo_infantil']

    resultado = {'timestamp': datetime.now().isoformat(), 'origem': args.csv,
                 'n_buscas': len(buscas), 'n_videos': len(videos)}

    # ---- distribuição por estrato ----
    print('=' * 78)
    print('DISTRIBUIÇÃO DO SCORE POR ESTRATO')
    print('=' * 78)
    print(f'{"estrato / termo":<34} {"n":>3} {"média":>7} {"mín":>7} {"máx":>7} {"amplitude":>10}')
    print('-' * 78)
    por_termo = {}
    for b in buscas:
        s = [v['score_media'] for v in b['videos']]
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

    # ---- separação entre estratos ----
    sa = [v['score_media'] for v in A]
    sb = [v['score_media'] for v in B]
    u, p = mannwhitneyu(sa, sb, alternative='greater')
    auc, lo, hi = auc_com_ic(sa, sb, rng)
    resultado['separacao_media'] = {'mannwhitney_u': float(u), 'p': float(p),
                                    'auc': auc, 'auc_ic95': [lo, hi],
                                    'diferenca_medias': float(np.mean(sa) - np.mean(sb))}

    sa_min = [v['score_minimo'] for v in A]
    sb_min = [v['score_minimo'] for v in B]
    auc_m, lo_m, hi_m = auc_com_ic(sa_min, sb_min, rng)
    u_m, p_m = mannwhitneyu(sa_min, sb_min, alternative='greater')
    resultado['separacao_minimo'] = {'mannwhitney_u': float(u_m), 'p': float(p_m),
                                     'auc': auc_m, 'auc_ic95': [lo_m, hi_m]}

    print('\n' + '=' * 78)
    print('SEPARAÇÃO ENTRE ESTRATOS (A deveria pontuar acima de B)')
    print('=' * 78)
    print(f'  diferença de médias:  {np.mean(sa) - np.mean(sb):+.4f}')
    print(f'  Mann-Whitney:         U = {u:.1f}, p = {p:.5f}')
    print(f'  AUC (média das 3):    {auc:.3f}  IC95% [{lo:.3f}, {hi:.3f}]')
    print(f'  AUC (mínimo das 3):   {auc_m:.3f}  IC95% [{lo_m:.3f}, {hi_m:.3f}]  '
          f'(p = {p_m:.5f})')

    # ---- detecção por limiar ----
    print('\n' + '=' * 78)
    print('DETECÇÃO POR LIMIAR')
    print('=' * 78)
    print(f'{"limiar":>8} {"sinalizados em B":>18} {"falso alarme em A":>19}')
    print('-' * 78)
    tabela_limiar = []
    for lim in LIMIARES:
        nb = sum(1 for s in sb if s < lim)
        na = sum(1 for s in sa if s < lim)
        tabela_limiar.append({'limiar': lim, 'detectados_b': nb, 'total_b': len(sb),
                              'falsos_a': na, 'total_a': len(sa)})
        print(f'{lim:>8.2f} {f"{nb}/{len(sb)} ({nb/len(sb):.0%})":>18} '
              f'{f"{na}/{len(sa)} ({na/len(sa):.0%})":>19}')
    resultado['deteccao_por_limiar'] = tabela_limiar

    # ---- taxonomia das falhas no estrato B ----
    LIMIAR_REF = 0.85
    nao_detectados = [v for v in B if v['score_media'] >= LIMIAR_REF]
    falha_amostragem = [v for v in nao_detectados if not v['tem_frase_toxica']]
    falha_modelo = [v for v in nao_detectados if v['tem_frase_toxica']]

    print('\n' + '=' * 78)
    print(f'FALHAS NO ESTRATO B (limiar {LIMIAR_REF:.2f})')
    print('=' * 78)
    print(f'  não sinalizados: {len(nao_detectados)} de {len(B)}')
    print(f'    por amostragem (nenhuma frase amostrada era tóxica): {len(falha_amostragem)}')
    print(f'    pelo classificador (havia frase tóxica não reconhecida): {len(falha_modelo)}')
    for v in nao_detectados:
        tipo = 'amostragem' if not v['tem_frase_toxica'] else 'classificador'
        print(f'\n    [{tipo}] {v["titulo"][:62]}')
        for t, s, c in zip(v['frases'].values(), v['scores_frases'], v['classes_frases']):
            print(f'       {s:.3f} ({c[:7]:<7}) {t[:66]}')
    resultado['falhas_estrato_b'] = {
        'limiar': LIMIAR_REF, 'nao_detectados': len(nao_detectados), 'total': len(B),
        'falha_amostragem': len(falha_amostragem), 'falha_modelo': len(falha_modelo),
        'detalhe': [{'titulo': v['titulo'], 'termo': v['termo'],
                     'score': v['score_media'],
                     'tipo': 'amostragem' if not v['tem_frase_toxica'] else 'classificador',
                     'frases': list(v['frases'].values()),
                     'classes': v['classes_frases']} for v in nao_detectados],
    }

    # ---- falsos alarmes no estrato A ----
    fa = [v for v in A if v['score_media'] < LIMIAR_REF]
    print(f'\n  falsos alarmes no estrato A: {len(fa)} de {len(A)}')
    for v in fa:
        print(f'    {v["score_media"]:.3f}  {v["titulo"][:60]}')
        for t, s, c in zip(v['frases'].values(), v['scores_frases'], v['classes_frases']):
            print(f'       {s:.3f} ({c[:7]:<7}) {t[:66]}')
    resultado['falsos_alarme_a'] = [{'titulo': v['titulo'], 'score': v['score_media'],
                                     'frases': list(v['frases'].values()),
                                     'classes': v['classes_frases']} for v in fa]

    # ---- deslocamento no ranqueamento ----
    print('\n' + '=' * 78)
    print('MAIORES DESLOCAMENTOS')
    print('=' * 78)
    deslocamentos = []
    for b in buscas:
        ordem = sorted(b['videos'], key=lambda v: -v['score_media'])
        for novo, v in enumerate(ordem, 1):
            deslocamentos.append({'termo': b['termo'], 'estrato': b['estrato'],
                                  'titulo': v['titulo'], 'de': v['posicao_youtube'],
                                  'para': novo, 'delta': novo - v['posicao_youtube'],
                                  'score': v['score_media']})
    for d in sorted(deslocamentos, key=lambda x: -abs(x['delta']))[:6]:
        print(f'  {d["de"]:>2} -> {d["para"]:>2} ({d["delta"]:+d})  score {d["score"]:.3f}  '
              f'{d["titulo"][:50]}')
    resultado['deslocamentos'] = deslocamentos

    destino = Path(project_root) / 'app' / 'nlp' / 'evaluation' / 'results' / \
        f'estratos_{datetime.now():%Y%m%d_%H%M%S}.json'
    with open(destino, 'w', encoding='utf-8') as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False, default=str)
    print(f'\nSalvo em: {destino}')

    gerar_figura(buscas, LIMIAR_REF)
    gerar_tabela(resultado, buscas)


def gerar_figura(buscas, limiar):
    """Dispersão dos scores por termo, com os dois estratos em cores distintas."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    COR_A, COR_B = '#2ecc71', '#e74c3c'
    rng = np.random.default_rng(3)

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    rotulos = []
    for i, b in enumerate(buscas):
        cor = COR_A if b['estrato'] == 'A_responsavel' else COR_B
        s = [v['score_media'] for v in b['videos']]
        x = i + rng.uniform(-0.16, 0.16, len(s))
        ax.scatter(x, s, s=52, color=cor, alpha=0.85, edgecolor='white', linewidth=1.1,
                   zorder=3)
        ax.plot([i - 0.30, i + 0.30], [np.mean(s)] * 2, color='#2c3e50',
                linewidth=1.8, zorder=4)
        termo = b['termo']
        rotulos.append(termo if len(termo) <= 22 else termo[:20] + '…')

    ax.axhline(limiar, color='#7f8c8d', linestyle='--', linewidth=1.3, zorder=2)
    if IDIOMA == 'en':
        texto_limiar = f'threshold {limiar:.2f}'
        rot_a, rot_b = 'A — caregiver query', 'B — child appeal'
        rot_y = 'Video safety score'
    else:
        texto_limiar = f'limiar {limiar:.2f}'.replace('.', ',')
        rot_a, rot_b = 'A — busca do responsável', 'B — apelo infantil'
        rot_y = 'Score de segurança do vídeo'
    ax.text(len(buscas) - 0.52, limiar + 0.012, texto_limiar,
            fontsize=9, color='#7f8c8d')
    ax.axvline(1.5, color='#bdc3c7', linewidth=1.0, zorder=1)
    ax.text(0.5, 0.055, rot_a, ha='center', fontsize=9.5,
            color=COR_A, fontweight='bold')
    ax.text(2.5, 0.055, rot_b, ha='center', fontsize=9.5,
            color=COR_B, fontweight='bold')

    ax.set_xticks(range(len(buscas)))
    ax.set_xticklabels(rotulos, fontsize=9)
    ax.set_ylabel(rot_y, fontsize=10)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.5, len(buscas) - 0.5)
    ax.grid(axis='y', alpha=0.25, linestyle=':')
    ax.set_axisbelow(True)
    for lado in ('top', 'right'):
        ax.spines[lado].set_visible(False)

    nome_arquivo = 'fig11-estratos.png' if IDIOMA == 'en' else 'fig11-estratos-pt.png'
    caminho = Path(project_root).parent / 'latex' / 'figuras' / nome_arquivo
    fig.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Figura: {caminho}')


def _dec(texto):
    """Aplica o separador decimal do idioma a uma linha ja formatada."""
    return texto if IDIOMA == 'en' else texto.replace('.', '{,}')


def gerar_tabela(resultado, buscas):
    """Tabela LaTeX com a distribuição do score por termo de busca."""
    if IDIOMA == 'en':
        legenda = (r'\caption{Distribution of the safety score across the four queries of '
                   r'the case study, ten videos each. In stratum A the desirable outcome '
                   r'is that scores be high and close to one another; in stratum B, that '
                   r'they be low. Queries are given verbatim in Brazilian Portuguese, '
                   r'since they are inputs to the system.}')
        cabecalho = r'Stratum & Query & $n$ & Mean & Minimum & Range \\'
        agregado = 'aggregate'
    else:
        legenda = (r'\caption{Distribuição do \emph{score} de segurança nas quatro buscas do '
                   r'estudo de caso, com dez vídeos cada. No estrato A o desejável é que os '
                   r'\emph{scores} sejam altos e próximos entre si; no B, que sejam baixos.}')
        cabecalho = r'Estrato & Termo de busca & $n$ & Média & Mínimo & Amplitude \\'
        agregado = 'agregado'
    linhas = [r'\begin{table}[!t]', r'\centering', legenda,
              r'\label{tab:estratos}',
              r'\begin{tabular}{llcccc}', r'\hline',
              cabecalho, r'\hline']
    nomes = {'A_responsavel': 'A', 'B_apelo_infantil': 'B'}
    for b in buscas:
        d = resultado['por_termo'][b['termo']]
        termo = b['termo'].replace('&', r'\&')
        linhas.append(_dec(
            f"{nomes[b['estrato']]} & \\emph{{{termo}}} & {d['n']} & "
            f"{d['media']:.3f} & {d['min']:.3f} & {d['amplitude']:.3f} \\\\"))
    linhas.append(r'\hline')
    for chave, rot in (('estrato_A_responsavel', 'A'), ('estrato_B_apelo_infantil', 'B')):
        d = resultado[chave]
        linhas.append(_dec(
            f"\\textbf{{{rot}}} & \\emph{{{agregado}}} & {d['n']} & "
            f"\\textbf{{{d['media']:.3f}}} & {d['min']:.3f} & {d['amplitude']:.3f} \\\\"))
    linhas += [r'\hline', r'\end{tabular}', r'\end{table}', '']
    saida = Path(project_root).parent / 'latex' / 'tabelas' / 'tab-estratos.tex'
    saida.write_text('\n'.join(linhas), encoding='utf-8')
    print(f'Tabela LaTeX: {saida}')


if __name__ == '__main__':
    main()
