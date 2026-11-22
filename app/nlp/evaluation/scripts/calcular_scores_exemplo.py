"""
Recalcula o exemplo numérico do \\emph{score de segurança} usado no artigo,
com o modelo de toxicidade recém-treinado.

Usa as frases já em cache (app/cache/cache_pernalonga.json), que são as três
sentenças representativas — início, meio e fim — extraídas de cada vídeo pelo
protótipo. Sem isso, o exemplo numérico do artigo ficaria com as probabilidades
do modelo antigo.

Saída: evaluation/results/artigo/scores_exemplo.json

Uso:
    python -m app.nlp.evaluation.scripts.calcular_scores_exemplo
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp import experimento  # noqa: E402

from app.nlp.models.bertimbau_toxicity import BertimbauToxicity

# Mesmos pesos de penalização do filtro em produção (app/filters/toxicity.py)
PESOS = {'Nenhuma': 0.0, 'Leve': 0.5, 'Severa': 1.0}

CACHE = os.path.join(project_root, 'app', 'cache', 'cache_pernalonga.json')
OUTPUT_DIR = str(experimento.dir_resultados('artigo'))


def modelo_mais_recente() -> str:
    """Modelo de referência da rodada (ver app/nlp/experimento.py)."""
    return str(experimento.caminho_modelo_referencia())


def distribuicao_por_classe(modelo, batch_size=64):
    """
    Distribuição do \\emph{score} de segurança sobre o conjunto de teste,
    separada pela classe anotada.

    É a evidência de que o \\emph{score} ordena conteúdo: se as três classes
    produzem distribuições sobrepostas, a reordenação não separa nada. Um punhado
    de vídeos de demonstração não sustenta essa afirmação; as 302 frases do teste
    retido, sim.
    """
    import torch
    from app.nlp.datasets.prepare_data_toxicity import get_data_for_cv_and_test

    _, (textos, rotulos) = get_data_for_cv_and_test(test_size=0.20)
    pesos = [PESOS[c] for c in ('Nenhuma', 'Leve', 'Severa')]

    scores = []
    modelo.model.eval()
    with torch.no_grad():
        for i in range(0, len(textos), batch_size):
            entradas = modelo.tokenizer(
                textos[i:i + batch_size], padding='max_length', truncation=True,
                max_length=modelo.max_length, return_tensors='pt'
            )
            entradas = {k: v.to(modelo.device) for k, v in entradas.items()}
            probs = torch.softmax(modelo.model(**entradas).logits, dim=-1).cpu().numpy()
            scores.extend((1.0 - probs @ pesos).tolist())

    por_classe = {}
    for indice, nome in enumerate(('Nenhuma', 'Leve', 'Severa')):
        valores = [s for s, r in zip(scores, rotulos) if r == indice]
        por_classe[nome] = {
            'n': len(valores),
            'media': round(float(np.mean(valores)), 4),
            'mediana': round(float(np.median(valores)), 4),
            'p25': round(float(np.percentile(valores, 25)), 4),
            'p75': round(float(np.percentile(valores, 75)), 4),
            'min': round(float(np.min(valores)), 4),
            'max': round(float(np.max(valores)), 4),
            'scores': [round(float(v), 4) for v in valores],
        }
        print(f"  {nome:8}: n={len(valores):3} | média={por_classe[nome]['media']:.3f} "
              f"| mediana={por_classe[nome]['mediana']:.3f} "
              f"| faixa {por_classe[nome]['min']:.3f}–{por_classe[nome]['max']:.3f}")

    return por_classe


def main():
    caminho_modelo = modelo_mais_recente()
    print(f"Modelo: {caminho_modelo}\n")
    modelo = BertimbauToxicity(model_path=caminho_modelo)

    with open(CACHE, encoding='utf-8') as f:
        videos = json.load(f)

    saida = {'modelo': os.path.basename(caminho_modelo), 'pesos': PESOS, 'videos': {}}

    for titulo, frases in videos.items():
        detalhes = []
        for parte in ('start', 'middle', 'end'):
            texto = frases.get(parte, '')
            if not texto.strip():
                continue
            resultado = modelo.predict_toxicity(texto, return_probabilities=True)
            probs = resultado['probabilities']
            penalizacao = sum(probs[c] * PESOS[c] for c in PESOS)
            detalhes.append({
                'parte': parte,
                'texto': texto,
                'probabilidades': {c: round(probs[c], 4) for c in PESOS},
                'classe_predita': resultado['predicted_label'],
                'penalizacao': round(penalizacao, 4),
                'score_frase': round(1 - penalizacao, 4),
            })

        score_final = sum(d['score_frase'] for d in detalhes) / len(detalhes)
        saida['videos'][titulo] = {
            'frases': detalhes,
            'score_toxicidade': round(score_final, 4),
            'score_toxicidade_pct': round(score_final * 100),
        }

        print(f"=== {titulo}")
        for d in detalhes:
            p = d['probabilidades']
            print(f"  [{d['parte']:>6}] Nenhuma={p['Nenhuma']:.4f} "
                  f"Leve={p['Leve']:.4f} Severa={p['Severa']:.4f} "
                  f"-> Pen={d['penalizacao']:.4f} P={d['score_frase']:.4f}")
        print(f"  SCORE DE TOXICIDADE: {score_final:.4f} ({score_final*100:.0f}%)\n")

    print("Distribuição do score por classe anotada (teste retido):")
    saida['distribuicao_por_classe'] = distribuicao_por_classe(modelo)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    destino = os.path.join(OUTPUT_DIR, 'scores_exemplo.json')
    with open(destino, 'w', encoding='utf-8') as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)
    print(f"Salvo em: {destino}")


if __name__ == '__main__':
    main()
