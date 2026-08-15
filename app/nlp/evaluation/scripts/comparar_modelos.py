"""
Comparação pareada entre modelos no conjunto de teste retido.

Com 302 amostras de teste, diferenças de dois ou três pontos de F1 podem ser
ruído amostral. Este script recalcula as predições de cada modelo sobre
exatamente as mesmas frases e compara os pares por *bootstrap* pareado: reamostra
as frases com reposição, recalcula a diferença de F1 macro em cada reamostragem
e reporta o intervalo de confiança dessa diferença. O intervalo contendo zero
significa que o teste não distingue os dois modelos.

Também roda o teste de McNemar sobre acertos/erros, que responde a uma pergunta
mais estrita: os dois modelos erram nas mesmas frases?

Uso:
    python -m app.nlp.evaluation.scripts.comparar_modelos
"""

import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
from sklearn.metrics import f1_score

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('app').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

N_BOOTSTRAP = 5000
RANDOM_STATE = 42
CLASSES = ['Nenhuma', 'Leve', 'Severa']


def predicoes_bertimbau(X_test):
    """Predições do modelo ajustado mais recente."""
    from pathlib import Path
    from app.nlp.models.bertimbau_toxicity import BertimbauToxicity
    import torch

    trained = Path(project_root) / 'app' / 'nlp' / 'models' / 'trained'
    candidatos = [p for p in trained.glob('TOX_*') if (p / 'config.json').exists()]
    caminho = max(candidatos, key=lambda p: p.stat().st_mtime)
    logger.info(f"BERTimbau ajustado: {caminho.name}")

    modelo = BertimbauToxicity(model_path=str(caminho))
    modelo.model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 64):
            entradas = modelo.tokenizer(X_test[i:i + 64], padding='max_length',
                                        truncation=True, max_length=modelo.max_length,
                                        return_tensors='pt')
            entradas = {k: v.to(modelo.device) for k, v in entradas.items()}
            preds.extend(torch.argmax(modelo.model(**entradas).logits, dim=-1).cpu().numpy().tolist())

    del modelo
    torch.cuda.empty_cache()
    return preds


def predicoes_sonda(X_train, y_train, X_test):
    """Predições da sonda linear sobre representações congeladas."""
    from sklearn.linear_model import LogisticRegression
    from app.nlp.training.train_toxicity_sem_finetuning import extrair_embeddings

    E_train = extrair_embeddings(list(X_train))
    E_test = extrair_embeddings(list(X_test))
    clf = LogisticRegression(max_iter=3000, class_weight='balanced',
                             random_state=RANDOM_STATE, multi_class='multinomial')
    clf.fit(E_train, y_train)
    return clf.predict(E_test).tolist()


def predicoes_baselines(X_train, y_train, X_test):
    """Predições de cada baseline TF-IDF, treinada no mesmo conjunto."""
    from sklearn.base import clone
    from app.nlp.training.train_toxicity_baselines import BASELINES

    saida = {}
    for nome, pipeline in BASELINES.items():
        modelo = clone(pipeline).fit(np.array(X_train), np.array(y_train))
        saida[nome] = modelo.predict(np.array(X_test)).tolist()
    return saida


def bootstrap_pareado(y_true, pred_a, pred_b, n=N_BOOTSTRAP):
    """
    Intervalo de confiança de 95% para a diferença de F1 macro (a - b).

    Reamostra os índices das frases, e não as predições, para que os dois
    modelos sejam sempre avaliados sobre o mesmo subconjunto — é isso que torna
    a comparação pareada e elimina a variação comum aos dois.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    y_true = np.array(y_true)
    pred_a = np.array(pred_a)
    pred_b = np.array(pred_b)

    observada = (f1_score(y_true, pred_a, average='macro', zero_division=0)
                 - f1_score(y_true, pred_b, average='macro', zero_division=0))

    diffs = np.empty(n)
    indices = np.arange(len(y_true))
    for i in range(n):
        amostra = rng.choice(indices, size=len(indices), replace=True)
        diffs[i] = (f1_score(y_true[amostra], pred_a[amostra], average='macro', zero_division=0)
                    - f1_score(y_true[amostra], pred_b[amostra], average='macro', zero_division=0))

    baixo, alto = np.percentile(diffs, [2.5, 97.5])
    return {
        'diferenca_observada': float(observada),
        'ic95_baixo': float(baixo),
        'ic95_alto': float(alto),
        'p_bootstrap': float(min((diffs <= 0).mean(), (diffs >= 0).mean()) * 2),
        'significativo': bool(baixo > 0 or alto < 0),
    }


def mcnemar(y_true, pred_a, pred_b):
    """Teste de McNemar sobre acertos e erros discordantes."""
    from scipy.stats import binomtest

    y_true = np.array(y_true)
    acerta_a = np.array(pred_a) == y_true
    acerta_b = np.array(pred_b) == y_true

    so_a = int((acerta_a & ~acerta_b).sum())
    so_b = int((~acerta_a & acerta_b).sum())
    if so_a + so_b == 0:
        return {'so_a': 0, 'so_b': 0, 'p_valor': 1.0}

    p = binomtest(so_a, so_a + so_b, 0.5).pvalue
    return {'so_a': so_a, 'so_b': so_b, 'p_valor': float(p)}


def main():
    from app.nlp.datasets.prepare_data_toxicity import get_data_for_cv_and_test

    (X_train, y_train), (X_test, y_test) = get_data_for_cv_and_test(test_size=0.20)
    logger.info(f"Teste: {len(X_test)} frases")

    predicoes = {'BERTimbau (ajuste fino)': predicoes_bertimbau(list(X_test))}
    logger.info("Sonda linear...")
    predicoes['Sonda linear (congelado)'] = predicoes_sonda(X_train, y_train, list(X_test))
    logger.info("Baselines TF-IDF...")
    predicoes.update(predicoes_baselines(X_train, y_train, list(X_test)))

    print("\n" + "=" * 78)
    print("F1 MACRO NO TESTE RETIDO")
    print("=" * 78)
    f1s = {n: f1_score(y_test, p, average='macro', zero_division=0)
           for n, p in predicoes.items()}
    for nome, f1 in sorted(f1s.items(), key=lambda kv: -kv[1]):
        print(f"  {nome:<34} {f1:.4f}")

    referencia = 'BERTimbau (ajuste fino)'
    comparacoes = {}
    print("\n" + "=" * 78)
    print(f"COMPARAÇÃO PAREADA CONTRA {referencia} (bootstrap, 95%)")
    print("=" * 78)
    print(f"  {'Contra':<30} {'Δ F1':>8} {'IC 95%':>20} {'McNemar p':>11}  Conclusão")
    print("-" * 78)

    for nome in predicoes:
        if nome == referencia:
            continue
        bs = bootstrap_pareado(y_test, predicoes[referencia], predicoes[nome])
        mc = mcnemar(y_test, predicoes[referencia], predicoes[nome])
        comparacoes[nome] = {'bootstrap': bs, 'mcnemar': mc}
        veredito = 'diferença detectada' if bs['significativo'] else 'indistinguível'
        ic = f"[{bs['ic95_baixo']:+.3f}, {bs['ic95_alto']:+.3f}]"
        print(f"  {nome:<30} {bs['diferenca_observada']:>+8.4f} {ic:>20} "
              f"{mc['p_valor']:>11.4f}  {veredito}")
    print("=" * 78)

    destino = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', f'comparacao_pareada_{datetime.now():%Y%m%d_%H%M%S}.json'
    )
    with open(destino, 'w', encoding='utf-8') as f:
        json.dump({'f1_macro_teste': f1s, 'comparacoes_vs_' + referencia: comparacoes,
                   'n_teste': len(y_test), 'n_bootstrap': N_BOOTSTRAP},
                  f, indent=2, ensure_ascii=False)
    print(f"\nSalvo em: {destino}")


if __name__ == '__main__':
    main()
