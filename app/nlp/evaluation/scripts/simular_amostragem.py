"""
Quanto do desempenho no nível do vídeo depende de quantas frases são amostradas?

O estudo de caso (`analisar_estratos.py`) mostra que as falhas do reordenador são
de *amostragem*: o vídeo contém conteúdo tóxico, mas nenhuma das três sentenças
sorteadas o contém. Um estudo de caso demonstra que o fenômeno ocorre; não mede
sua magnitude, porque a amostra de buscas é pequena e escolhida propositalmente.

Esta simulação mede a magnitude com poder estatístico adequado, usando as 302
sentenças do teste retido como população de amostragem. Constrói-se um vídeo
sintético sorteando k sentenças, cada uma tóxica com probabilidade f, e mede-se
a taxa de detecção com o limiar calibrado para 5% de falso alarme em vídeos
inteiramente apropriados — de modo que todas as regras sejam comparadas no mesmo
ponto de operação.

Compara quatro regras de agregação das k sentenças num único score de vídeo,
sendo a média a definida na Equação do artigo.

Uso:
    python -m app.nlp.evaluation.scripts.simular_amostragem
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

logging.disable(logging.INFO)

PESOS = {'Nenhuma': 0.0, 'Leve': 0.5, 'Severa': 1.0}
SEED = 7
N_SIM = 30000
KS = (3, 5, 8, 12)
FRACOES = (0.10, 0.20)
FALSO_ALARME_ALVO = 5  # percentil, em vídeo 100% apropriado

REGRAS = {
    'media': lambda s: s.mean(axis=1),
    'minimo': lambda s: s.min(axis=1),
    'min_ponderado': lambda s: 0.7 * s.min(axis=1) + 0.3 * s.mean(axis=1),
    'duas_piores': lambda s: np.sort(s, axis=1)[:, :2].mean(axis=1),
}

ROTULOS = {'media': 'Média das $k$ (Eq. atual)', 'minimo': 'Mínimo das $k$',
           'min_ponderado': '$0{,}7\\cdot$mín $+\\ 0{,}3\\cdot$média',
           'duas_piores': 'Média das 2 piores'}


def main():
    from app.nlp.datasets.prepare_data_toxicity import get_data_for_cv_and_test
    from app.nlp.models.bertimbau_toxicity import BertimbauToxicity

    trained = Path(project_root) / 'app' / 'nlp' / 'models' / 'trained'
    caminho = max([p for p in trained.glob('TOX_*') if (p / 'config.json').exists()],
                  key=lambda p: p.stat().st_mtime)
    print(f'Modelo: {caminho.name}')
    modelo = BertimbauToxicity(model_path=str(caminho))

    (_, _), (X_test, y_test) = get_data_for_cv_and_test()
    scores = []
    for t in X_test:
        probs = modelo.predict_toxicity(t, return_probabilities=True)['probabilities']
        scores.append(1.0 - sum(probs[c] * PESOS[c] for c in PESOS))
    scores, y = np.array(scores), np.array(y_test)
    idx = {c: np.where(y == c)[0] for c in (0, 1, 2)}
    print(f'População: {len(X_test)} sentenças de teste '
          f'({len(idx[0])} Nenhuma, {len(idx[1])} Leve, {len(idx[2])} Severa)\n')

    rng = np.random.default_rng(SEED)

    def video_seguro(k):
        return scores[rng.choice(idx[0], size=(N_SIM, k))]

    def video_com_fracao(k, f):
        """Cada uma das k sentenças é Severa com probabilidade f."""
        toxica = rng.random((N_SIM, k)) < f
        return np.where(toxica,
                        scores[rng.choice(idx[2], size=(N_SIM, k))],
                        scores[rng.choice(idx[0], size=(N_SIM, k))])

    resultado = {'timestamp': datetime.now().isoformat(), 'n_simulacoes': N_SIM,
                 'falso_alarme_alvo_pct': FALSO_ALARME_ALVO, 'deteccao': {}}

    for f in FRACOES:
        print(f'--- vídeo com {f:.0%} de conteúdo severamente tóxico ---')
        print(f'{"agregação":<26}' + ''.join(f'{f"k={k}":>8}' for k in KS))
        resultado['deteccao'][f'{f:.2f}'] = {}
        for nome, regra in REGRAS.items():
            linha, vals = f'{nome:<26}', {}
            for k in KS:
                lim = float(np.percentile(regra(video_seguro(k)), FALSO_ALARME_ALVO))
                det = float((regra(video_com_fracao(k, f)) < lim).mean())
                vals[k] = {'limiar': lim, 'deteccao': det}
                linha += f'{det:>7.0%} '
            resultado['deteccao'][f'{f:.2f}'][nome] = vals
            print(linha)
        print()

    # Dispersão em vídeo inteiramente apropriado — fundamenta a banda de tolerância
    print('--- dispersão do score em vídeo 100% apropriado (regra: média) ---')
    resultado['dispersao_video_seguro'] = {}
    for k in KS:
        s = REGRAS['media'](video_seguro(k))
        dif = np.abs(s[:N_SIM // 2] - s[N_SIM // 2:])
        resultado['dispersao_video_seguro'][k] = {
            'media': float(s.mean()), 'desvio': float(s.std()),
            'p5': float(np.percentile(s, 5)),
            'diferenca_mediana_entre_videos': float(np.median(dif)),
        }
        print(f'  k={k:<3} média {s.mean():.4f}  dp {s.std():.4f}  '
              f'|diferença| mediana entre dois vídeos apropriados: {np.median(dif):.4f}')

    destino = Path(project_root) / 'app' / 'nlp' / 'evaluation' / 'results' / \
        f'simulacao_amostragem_{datetime.now():%Y%m%d_%H%M%S}.json'
    with open(destino, 'w', encoding='utf-8') as fh:
        json.dump(resultado, fh, indent=2, ensure_ascii=False)
    print(f'\nSalvo em: {destino}')

    # Tabela LaTeX para o artigo
    f_ref = '0.20'
    tex = [r'\begin{table}[!t]', r'\centering',
           r'\caption{Detecção simulada de vídeo com 20\% de conteúdo severamente '
           r'tóxico, em função do número $k$ de sentenças amostradas e da regra de '
           r'agregação. Cada regra é calibrada para 5\% de falso alarme sobre vídeos '
           r'inteiramente apropriados, de modo que as colunas sejam comparáveis. '
           r'População de amostragem: as 302 sentenças do teste retido.}',
           r'\label{tab:simulacao}',
           r'\begin{tabular}{l' + 'c' * len(KS) + '}', r'\hline',
           'Agregação & ' + ' & '.join(f'$k={k}$' for k in KS) + r' \\', r'\hline']
    for nome in REGRAS:
        vals = resultado['deteccao'][f_ref][nome]
        celulas = [f"{vals[k]['deteccao']:.0%}".replace('%', r'\%') for k in KS]
        linha = f'{ROTULOS[nome]} & ' + ' & '.join(celulas) + r' \\'
        if nome == 'min_ponderado':
            linha = r'\textbf{' + ROTULOS[nome] + '} & ' + \
                ' & '.join(r'\textbf{' + c + '}' for c in celulas) + r' \\'
        tex.append(linha)
    tex += [r'\hline', r'\end{tabular}', r'\end{table}', '']
    saida = Path(project_root).parent / 'latex' / 'tabelas' / 'tab-simulacao.tex'
    saida.write_text('\n'.join(tex), encoding='utf-8')
    print(f'Tabela LaTeX: {saida}')


if __name__ == '__main__':
    main()
