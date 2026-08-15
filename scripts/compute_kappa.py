"""
Cálculo de concordância entre rotuladores — Fleiss Kappa (item A2).
--------------------------------------------------------------------
Lê um CSV com os votos REAIS dos três rotuladores e calcula a concordância
inter-anotadores. NÃO gera votos: opera apenas sobre os dados fornecidos.

CSV de entrada esperado (padrão em src/data/votos_rotuladores.csv):
    id, frase, rot1, rot2, rot3
onde rot1/rot2/rot3 ∈ {0,1,2} (0=Nenhuma, 1=Leve, 2=Severa).
Linhas com algum voto em branco são ignoradas no cálculo.

Uso:
    python src/scripts/compute_kappa.py
    python src/scripts/compute_kappa.py --csv src/data/votos_rotuladores.csv

Saída: Fleiss Kappa, concordância percentual, distribuição de unanimidade e o
rótulo final por moda (desempate = rot3, a psicóloga). Reporte no artigo
SOMENTE o número que sair daqui a partir de votos reais.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

RATER_COLS = ['rot1', 'rot2', 'rot3']
CATEGORIES = [0, 1, 2]


def fleiss_kappa(counts: np.ndarray) -> float:
    """
    Fleiss Kappa a partir da matriz N×k de contagens (n_ij = nº de rotuladores
    que atribuíram o item i à categoria j). Cada linha soma n (nº de rotuladores).
    """
    N, k = counts.shape
    n = counts.sum(axis=1)
    if not np.all(n == n[0]):
        raise ValueError("Todos os itens devem ter o mesmo número de rotuladores.")
    n = int(n[0])
    if n < 2:
        raise ValueError("Fleiss Kappa exige ao menos 2 rotuladores por item.")

    # Concordância por item
    P_i = (np.sum(counts ** 2, axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()

    # Proporção de atribuições por categoria
    p_j = counts.sum(axis=0) / (N * n)
    P_e = np.sum(p_j ** 2)

    if np.isclose(P_e, 1.0):
        return 1.0
    return (P_bar - P_e) / (1.0 - P_e)


def rotulo_por_moda(votos):
    """Moda dos 3 votos; empate total (3 distintos) → voto de rot3 (psicóloga)."""
    valores, contagens = np.unique(votos, return_counts=True)
    maximo = contagens.max()
    candidatos = valores[contagens == maximo]
    if len(candidatos) == 1:
        return int(candidatos[0])
    return int(votos[2])  # desempate: rot3


def main():
    parser = argparse.ArgumentParser(description="Fleiss Kappa dos rotuladores (dados reais)")
    default_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'votos_rotuladores.csv')
    parser.add_argument('--csv', default=default_csv, help="CSV com id,frase,rot1,rot2,rot3")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERRO: arquivo não encontrado: {args.csv}")
        print("Preencha a planilha de anotação com os votos REAIS dos três rotuladores.")
        sys.exit(1)

    try:
        df = pd.read_csv(args.csv, sep=';')
        if 'rot1' not in df.columns and 'P1' in df.columns:
            # Mapear colunas do corpus_toxicidade.csv se for passado
            df = df.rename(columns={'P1': 'rot1', 'P2': 'rot2', 'A1': 'rot3'})
    except Exception:
        df = pd.read_csv(args.csv)

    faltando = [c for c in RATER_COLS if c not in df.columns]
    if faltando:
        print(f"ERRO: colunas ausentes no CSV: {faltando}")
        sys.exit(1)

    label_map = {'nenhuma': 0, 'leve': 1, 'severa': 2, 0: 0, 1: 1, 2: 2, '0': 0, '1': 1, '2': 2}
    for c in RATER_COLS:
        df[c] = df[c].astype(str).str.strip().str.lower().map(label_map)

    total = len(df)
    df = df.dropna(subset=RATER_COLS)
    validos = len(df)
    if validos == 0:
        print("ERRO: nenhuma linha com os três votos preenchidos. Nada a calcular.")
        sys.exit(1)

    votos = df[RATER_COLS].astype(int).to_numpy()

    # Matriz de contagens N×3
    counts = np.zeros((validos, len(CATEGORIES)), dtype=int)
    for i, linha in enumerate(votos):
        for v in linha:
            counts[i, CATEGORIES.index(v)] += 1

    kappa = fleiss_kappa(counts)

    # Concordância
    unanimes = int(np.sum([len(set(l)) == 1 for l in votos]))
    dissenso_1 = int(np.sum([len(set(l)) == 2 for l in votos]))
    todos_diferentes = int(np.sum([len(set(l)) == 3 for l in votos]))
    # Concordância média por par
    pares = [(0, 1), (0, 2), (1, 2)]
    acordo_par = np.mean([np.mean(votos[:, a] == votos[:, b]) for a, b in pares])

    # Rótulos finais por moda
    finais = np.array([rotulo_por_moda(l) for l in votos])
    dist_final = {c: int(np.sum(finais == c)) for c in CATEGORIES}

    def faixa(k):
        for lim, nome in [(0.0, 'Poor'), (0.20, 'Slight'), (0.40, 'Fair'),
                          (0.60, 'Moderate'), (0.80, 'Substantial'), (1.01, 'Almost perfect')]:
            if k < lim:
                return nome
        return 'Almost perfect'

    print("=" * 64)
    print("CONCORDÂNCIA ENTRE ROTULADORES — Fleiss Kappa")
    print("=" * 64)
    print(f"Arquivo: {args.csv}")
    print(f"Frases com os 3 votos: {validos}/{total}")
    print(f"\nFleiss Kappa: {kappa:.4f}  ({faixa(kappa)}, escala de Landis & Koch)")
    print(f"Concordância média por par: {acordo_par*100:.1f}%")
    print(f"Unânimes (3/3): {unanimes} ({unanimes/validos*100:.1f}%)")
    print(f"Dissenso de 1 rotulador: {dissenso_1} ({dissenso_1/validos*100:.1f}%)")
    print(f"Três votos distintos: {todos_diferentes} ({todos_diferentes/validos*100:.1f}%)")
    print(f"\nDistribuição do rótulo final (moda, desempate=rot3): {dist_final}")
    print("=" * 64)


if __name__ == "__main__":
    main()
