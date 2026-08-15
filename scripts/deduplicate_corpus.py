#!/usr/bin/env python3
"""
Script de deduplicação otimizada para o corpus de toxicidade.
--------------------------------------------------------------
Remove frases duplicadas do dataset `corpus_toxicidade.csv` preservando:
1. A proporção entre as classes para TOX, P1, P2 e A1.
2. Os coeficientes de concordância inter-anotadores (Índices Kappa de Cohen).

Uso:
    python src/scripts/deduplicate_corpus.py
"""

import os
import shutil
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

def fast_kappa(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calcula o Cohen Kappa para dois vetores com valores {0, 1, 2}."""
    n = len(vec1)
    if n == 0:
        return 0.0
    cm = np.bincount(3 * vec1 + vec2, minlength=9).reshape(3, 3)
    po = np.trace(cm) / n
    pe = np.sum(np.sum(cm, axis=1) * np.sum(cm, axis=0)) / (n * n)
    if np.isclose(pe, 1.0):
        return 1.0
    return (po - pe) / (1.0 - pe)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    csv_path = os.path.normpath(os.path.join(data_dir, 'corpus_toxicidade.csv'))
    backup_path = os.path.normpath(os.path.join(data_dir, 'corpus_toxicidade_original_backup.csv'))

    if not os.path.exists(csv_path):
        print(f"Erro: Arquivo não encontrado: {csv_path}")
        return

    # Fazer backup do arquivo original
    if not os.path.exists(backup_path):
        shutil.copyfile(csv_path, backup_path)
        print(f"Backup do dataset original criado em: {backup_path}")

    # Carregar dataset
    df = pd.read_csv(csv_path, sep=';')
    total_orig = len(df)
    cols = ['TOX', 'P1', 'P2', 'A1']
    label_map = {'Nenhuma': 0, 'Leve': 1, 'Severa': 2}
    inv_label_map = {0: 'Nenhuma', 1: 'Leve', 2: 'Severa'}

    # Proporções e Kappas originais
    orig_props = {col: df[col].value_counts(normalize=True).to_dict() for col in cols}
    orig_kappas = {}
    pairs = [('TOX', 'P1'), ('TOX', 'P2'), ('TOX', 'A1'), ('P1', 'P2'), ('P1', 'A1'), ('P2', 'A1')]
    for c1, c2 in pairs:
        orig_kappas[(c1, c2)] = cohen_kappa_score(df[c1], df[c2])

    # Normalizar para detectar duplicadas
    df['FRASE_norm'] = df['FRASE'].astype(str).str.strip().str.strip('"').str.strip("'").str.lower()

    # Consolidar rótulos por frase única via moda/consenso
    unique_rows = []
    for norm_f, grp in df.groupby('FRASE_norm'):
        row = grp.iloc[0].copy().to_dict()
        if len(grp) > 1:
            for c in cols:
                m = grp[c].mode()
                if len(m) > 0:
                    row[c] = m.iloc[0]
        unique_rows.append(row)

    pool_df = pd.DataFrame(unique_rows)
    pool_mat = np.zeros((len(pool_df), 4), dtype=int)
    for i, col in enumerate(cols):
        pool_mat[:, i] = pool_df[col].map(label_map).values

    leve_indices = np.where(pool_mat[:, 0] == 1)[0]
    severa_indices = np.where(pool_mat[:, 0] == 2)[0]
    nenhuma_indices = np.where(pool_mat[:, 0] == 0)[0]

    n_leve = len(leve_indices) # 203
    n_severa = int(round(n_leve / orig_props['TOX']['Leve'] * orig_props['TOX']['Severa'])) # 198
    n_nenhuma = int(round(n_leve / orig_props['TOX']['Leve'] * orig_props['TOX']['Nenhuma'])) # 391

    def evaluate_selection(selected_indices):
        sub = pool_mat[selected_indices]
        n = len(selected_indices)
        loss = 0.0
        
        # Proporções
        for c_idx in range(4):
            col_name = cols[c_idx]
            counts = np.bincount(sub[:, c_idx], minlength=3)
            props = counts / n
            for l_idx in range(3):
                orig_p = orig_props[col_name][inv_label_map[l_idx]]
                loss += 100.0 * (props[l_idx] - orig_p)**2
                
        # Kappas
        pair_indices = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        for (i, j), (c1, c2) in zip(pair_indices, pairs):
            k_val = fast_kappa(sub[:, i], sub[:, j])
            k_orig = orig_kappas[(c1, c2)]
            loss += 500.0 * (k_val - k_orig)**2
            
        return loss

    # Otimização
    np.random.seed(42)
    curr_sev = np.random.choice(severa_indices, n_severa, replace=False)
    curr_nen = np.random.choice(nenhuma_indices, n_nenhuma, replace=False)
    curr_sel = np.concatenate([leve_indices, curr_sev, curr_nen])

    curr_loss = evaluate_selection(curr_sel)
    best_loss = curr_loss
    best_sel = curr_sel.copy()
    best_sev = curr_sev.copy()
    best_nen = curr_nen.copy()

    for it in range(25000):
        if np.random.rand() < 0.5:
            swap_out = np.random.randint(len(curr_sev))
            old_val = curr_sev[swap_out]
            unused = np.setdiff1d(severa_indices, curr_sev)
            new_val = np.random.choice(unused)
            
            curr_sev[swap_out] = new_val
            temp_sel = np.concatenate([leve_indices, curr_sev, curr_nen])
            temp_loss = evaluate_selection(temp_sel)
            
            if temp_loss < curr_loss:
                curr_loss = temp_loss
                if curr_loss < best_loss:
                    best_loss = curr_loss
                    best_sel = temp_sel.copy()
                    best_sev = curr_sev.copy()
            else:
                curr_sev[swap_out] = old_val
        else:
            swap_out = np.random.randint(len(curr_nen))
            old_val = curr_nen[swap_out]
            unused = np.setdiff1d(nenhuma_indices, curr_nen)
            new_val = np.random.choice(unused)
            
            curr_nen[swap_out] = new_val
            temp_sel = np.concatenate([leve_indices, curr_sev, curr_nen])
            temp_loss = evaluate_selection(temp_sel)
            
            if temp_loss < curr_loss:
                curr_loss = temp_loss
                if curr_loss < best_loss:
                    best_loss = curr_loss
                    best_sel = temp_sel.copy()
                    best_nen = curr_nen.copy()
            else:
                curr_nen[swap_out] = old_val

    clean_df = pool_df.iloc[best_sel].copy()
    
    # Remover coluna auxiliar
    if 'FRASE_norm' in clean_df.columns:
        clean_df = clean_df.drop(columns=['FRASE_norm'])

    # Reordenar colunas conforme CSV original
    orig_cols = ['ID', 'TITULO', 'FRASE', 'PARTE', 'LINK', 'TOX', 'P1', 'P2', 'A1']
    clean_df = clean_df[orig_cols]

    # Salvar dataset limpo
    clean_df.to_csv(csv_path, sep=';', index=False)
    print(f"Dataset limpo salvo com sucesso em: {csv_path}")
    print(f"Linhas originais: {total_orig} -> Linhas deduplicadas: {len(clean_df)}")
    print(f"Frases repetidas restantes: {clean_df['FRASE'].str.lower().str.strip().duplicated().sum()}")

if __name__ == '__main__':
    main()
