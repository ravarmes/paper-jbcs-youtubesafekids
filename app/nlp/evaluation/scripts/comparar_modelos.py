"""
Comparação pareada entre modelos no conjunto de teste retido.

Com poucas centenas de frases de teste, diferenças de dois ou três pontos de F1
podem ser ruído amostral. Este script compara, sobre exatamente as mesmas frases,
o BERTimbau ajustado com cada um dos demais modelos -- a sonda linear sobre
representações congeladas, as 4 baselines TF-IDF fixas e, quando houver, as
baselines ajustadas por CV aninhada (train_toxicity_baselines_ajustados.py) --
e responde ao item M3d da revisão do JBCS: os testes têm de casar com as
afirmações, que são sobre F1 MACRO.

Teste principal: bootstrap pareado da diferença de F1 macro
    Reamostram-se as frases do teste com reposição (B = 5.000), recalcula-se o
    F1 macro dos dois modelos na MESMA reamostragem e guarda-se a diferença
    (BERTimbau - outro). Reporta-se:
      - IC de 95% percentil da diferença;
      - p bilateral = 2 x min(P*(dif <= 0), P*(dif >= 0)), a fração das
        reamostragens em que a diferença cruza o zero, dobrada (inversão do IC
        percentil: p < 0,05 <=> o IC de 95% exclui o zero). Foi assim que saiu o
        p = 0,40 da sonda linear na versão submetida: 0,3988 = 2 x 0,1994, isto
        é, 19,9% das 5.000 reamostragens tinham a sonda à frente;
      - p corrigido = 2 x min((n<=0 + 1)/(B + 1), (n>=0 + 1)/(B + 1)), que nunca
        é zero (com B = 5.000, o menor valor é ~4e-4) -- o p bruto acima vale
        0 sempre que nenhuma reamostragem cruza o zero, o que não é "p = 0";
      - correção de Holm sobre a família de comparações contra o BERTimbau
        (aplicada ao p corrigido), com o veredito a alfa = 0,05.
    Todas as comparações usam as mesmas 5.000 reamostragens (semente 42).

Bootstrap por vídeo (complementar ao anterior)
    A partição é por vídeo e as frases de um mesmo vídeo não são independentes;
    reamostrar frases subestima a variância. Aqui reamostram-se os VÍDEOS do
    teste, cada um com todas as suas frases. Mesmos campos e mesma correção.

McNemar (complementar)
    Teste exato (binomial) sobre as frases em que só um dos dois acerta. Compara
    ACURÁCIA pareada (contagens de discordância), não F1 macro; fica como
    complemento e é rotulado assim no JSON.

Predições do BERTimbau
    Lidas do CSV do treino final (modelo_final/probabilidades_teste_*.csv), as
    mesmas usadas nas demais análises: a coluna da semente de referência é o
    modelo de referência. Com as 5 sementes no CSV, a comparação é repetida para
    cada semente, para mostrar que o resultado não é próprio da semente 42. Só
    na falta do CSV o modelo de referência é carregado e roda a predição.

Uso:
    python -m app.nlp.evaluation.scripts.comparar_modelos
    python -m app.nlp.evaluation.scripts.comparar_modelos --resultados <pasta> --saida <pasta>
"""

import argparse
import glob
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp import experimento  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('app').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

N_BOOTSTRAP = 5000
RANDOM_STATE = 42
ALFA = 0.05
CLASSES = ['Nenhuma', 'Leve', 'Severa']
REFERENCIA = 'BERTimbau (ajuste fino)'
SONDA = 'Sonda linear (congelado)'

METODO_P = (
    'p_bootstrap = min(1, 2*min(P*(d<=0), P*(d>=0))), em que d é a diferença de F1 macro '
    '(BERTimbau - outro) em cada uma das B reamostragens pareadas: a fração das reamostragens '
    'em que a diferença cruza o zero, dobrada (inversão do IC percentil; p < 0,05 se e só se o '
    'IC de 95% exclui o zero). O p = 0,40 da sonda linear na versão submetida saiu dessa conta '
    '(0,3988 = 2 x 0,1994). p_bootstrap_corrigido = min(1, 2*min((#(d<=0)+1)/(B+1), '
    '(#(d>=0)+1)/(B+1))) evita p = 0 quando nenhuma reamostragem cruza o zero (Davison & '
    'Hinkley, 1997); a correção de Holm é aplicada ao p corrigido, sobre a família de '
    'comparações contra o BERTimbau.'
)


# ----------------------------------------------------------------------------
# Predições
# ----------------------------------------------------------------------------

def _mais_recente(padrao):
    arquivos = glob.glob(padrao)
    return max(arquivos, key=os.path.getmtime) if arquivos else None


def _alinhar(tabela, ids_teste, y_teste, origem):
    """
    Reordena um CSV de predições do teste pela ordem de `ids_teste`.

    Falha -- em vez de comparar frases trocadas -- se os id_sentenca não forem
    exatamente os do teste desta rodada ou se os rótulos gravados divergirem.
    """
    ids = set(int(i) for i in tabela['id_sentenca'])
    esperados = set(int(i) for i in ids_teste)
    if len(tabela) != len(ids_teste) or ids != esperados:
        raise ValueError(
            f"{origem}: os id_sentenca não são os do teste desta rodada "
            f"({len(tabela)} linhas, {len(ids & esperados)} em comum com as {len(esperados)} "
            f"frases do teste). O arquivo é de outra rodada ou de outro corpus?")
    tabela = tabela.set_index('id_sentenca').loc[list(ids_teste)].reset_index()
    if 'rotulo' in tabela.columns and not (tabela['rotulo'].values == np.asarray(y_teste)).all():
        raise ValueError(f"{origem}: rótulos divergem do corpus para os mesmos id_sentenca")
    return tabela


def predicoes_bertimbau_csv(pasta, ids_teste, y_teste):
    """
    Predições do BERTimbau por semente, lidas do CSV do treino final.

    Returns:
        dict com 'semente_referencia', 'por_semente' {semente: preds} e as fontes,
        ou None se não houver treino final gravado em `pasta`.
    """
    arq_json = _mais_recente(os.path.join(pasta, 'modelo_final', 'modelo_final_*.json'))
    if arq_json is None:
        return None
    with open(arq_json, encoding='utf-8') as f:
        final = json.load(f)
    arq_csv = os.path.join(pasta, 'modelo_final', f"probabilidades_teste_{final['timestamp']}.csv")
    if not os.path.exists(arq_csv):
        raise FileNotFoundError(f"{os.path.basename(arq_json)} sem o CSV correspondente "
                                f"({os.path.basename(arq_csv)})")
    tabela = _alinhar(pd.read_csv(arq_csv, sep=';', encoding='utf-8'), ids_teste, y_teste,
                      os.path.basename(arq_csv))

    por_semente = {}
    for semente in final.get('sementes', []):
        colunas = [f'p_{c}_s{semente}' for c in ('nenhuma', 'leve', 'severa')]
        if all(c in tabela.columns for c in colunas):
            por_semente[int(semente)] = tabela[colunas].values.argmax(axis=1).tolist()
    referencia = int(final.get('semente_referencia', next(iter(por_semente), 42)))
    if referencia not in por_semente:
        raise ValueError(f"{os.path.basename(arq_csv)} não tem as colunas da semente de "
                         f"referência {referencia}")
    return {
        'semente_referencia': referencia,
        'por_semente': por_semente,
        'arquivo_modelo_final': os.path.basename(arq_json),
        'arquivo_probabilidades': os.path.basename(arq_csv),
        'modelo_referencia': next((m.get('modelo') for m in final.get('teste_por_semente', [])
                                   if m.get('seed') == referencia), None),
    }


def predicoes_bertimbau_modelo(X_test):
    """Plano B: carrega o modelo de referência da rodada e prediz o teste."""
    try:
        caminho = experimento.caminho_modelo_referencia()
    except FileNotFoundError as e:
        raise SystemExit(
            f"Sem predições do BERTimbau: nenhum modelo_final/probabilidades_teste_*.csv na "
            f"pasta de resultados e nenhum modelo de referência ({e}). Rode antes "
            "train_toxicity_gridsearch --mode final desta rodada.")
    from app.nlp.models.bertimbau_toxicity import BertimbauToxicity
    import torch

    logger.info(f"BERTimbau ajustado (predição a partir do modelo): {caminho.name}")
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
    return preds, caminho.name


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
    """Predições de cada baseline TF-IDF fixa, treinada no mesmo conjunto."""
    from sklearn.base import clone
    from app.nlp.training.train_toxicity_baselines import BASELINES

    saida = {}
    for nome, pipeline in BASELINES.items():
        modelo = clone(pipeline).fit(np.array(X_train), np.array(y_train))
        saida[nome] = modelo.predict(np.array(X_test)).tolist()
    return saida


def predicoes_baselines_ajustadas(pasta, ids_teste, y_teste):
    """
    Predições das baselines ajustadas (CV aninhada), lidas do CSV mais recente.

    Returns:
        ({nome: preds}, nome do arquivo) ou ({}, None) se ainda não houver.
    """
    arq_json = _mais_recente(os.path.join(pasta, 'baselines_ajustados_*.json'))
    if arq_json is None:
        return {}, None
    with open(arq_json, encoding='utf-8') as f:
        dados = json.load(f)
    arq_csv = os.path.join(pasta, dados['arquivo_predicoes_teste'])
    tabela = _alinhar(pd.read_csv(arq_csv, sep=';', encoding='utf-8'), ids_teste, y_teste,
                      os.path.basename(arq_csv))
    saida = {}
    for chave, r in dados['resultados'].items():
        coluna = f'pred_{chave}'
        if coluna in tabela.columns:
            saida[r['nome']] = tabela[coluna].astype(int).tolist()
    return saida, os.path.basename(arq_csv)


# ----------------------------------------------------------------------------
# Estatística
# ----------------------------------------------------------------------------

def pesos_bootstrap(n, n_rep=N_BOOTSTRAP, semente=RANDOM_STATE):
    """
    Matriz [n_rep, n] de quantas vezes cada frase entra em cada reamostragem.

    Gera as reamostragens na mesma sequência da versão anterior do script
    (rng.choice repetido, semente 42), de modo que os ICs já publicados se
    reproduzam exatamente com as mesmas predições.
    """
    rng = np.random.RandomState(semente)
    indices = np.arange(n)
    W = np.empty((n_rep, n))
    for i in range(n_rep):
        W[i] = np.bincount(rng.choice(indices, size=n, replace=True), minlength=n)
    return W


def pesos_bootstrap_por_grupo(grupos, n_rep=N_BOOTSTRAP, semente=RANDOM_STATE):
    """Reamostra os grupos (vídeos) com reposição; cada frase herda o peso do seu vídeo."""
    rng = np.random.RandomState(semente)
    _, codigo = np.unique(np.asarray(grupos).astype(str), return_inverse=True)
    n_grupos = codigo.max() + 1
    Wg = np.empty((n_rep, n_grupos))
    for i in range(n_rep):
        Wg[i] = np.bincount(rng.randint(0, n_grupos, size=n_grupos), minlength=n_grupos)
    return Wg[:, codigo], int(n_grupos)


def f1_macro_ponderado(W, y_true, pred, n_classes=len(CLASSES)):
    """
    F1 macro de cada reamostragem (linha de W), sem laço.

    Equivale a f1_score(average='macro', zero_division=0) sobre a reamostragem:
    a média é tomada só sobre as classes presentes nela (em y_true ou na
    predição), como faz o scikit-learn sem `labels`.
    """
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    c = np.arange(n_classes)
    verd = y_true[:, None] == c
    prev = pred[:, None] == c
    tp = W @ (verd & prev).astype(float)
    fp = W @ (~verd & prev).astype(float)
    fn = W @ (verd & ~prev).astype(float)
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    presente = denom > 0
    return (f1 * presente).sum(axis=1) / np.maximum(presente.sum(axis=1), 1)


def bootstrap_pareado(W, y_true, pred_a, pred_b):
    """
    IC de 95% e p bilateral da diferença de F1 macro (a - b), dados os pesos W.

    Reamostra as frases (ou os vídeos, conforme W), e não as predições: os dois
    modelos são sempre avaliados sobre o mesmo subconjunto, o que torna a
    comparação pareada e elimina a variação comum aos dois.
    """
    observada = (f1_score(y_true, pred_a, average='macro', zero_division=0)
                 - f1_score(y_true, pred_b, average='macro', zero_division=0))
    diffs = f1_macro_ponderado(W, y_true, pred_a) - f1_macro_ponderado(W, y_true, pred_b)
    n = len(diffs)
    baixo, alto = np.percentile(diffs, [2.5, 97.5])
    n_le, n_ge = int((diffs <= 0).sum()), int((diffs >= 0).sum())
    return {
        'diferenca_observada': float(observada),
        'ic95_baixo': float(baixo),
        'ic95_alto': float(alto),
        'p_bootstrap': float(min(1.0, 2 * min(n_le / n, n_ge / n))),
        'p_bootstrap_corrigido': float(min(1.0, 2 * min((n_le + 1) / (n + 1), (n_ge + 1) / (n + 1)))),
        'n_reamostragens_dif_le_0': n_le,
        'n_reamostragens_dif_ge_0': n_ge,
        'significativo': bool(baixo > 0 or alto < 0),
    }


def holm(pvalores):
    """p ajustados por Holm (step-down), na ordem de entrada."""
    p = np.asarray(pvalores, dtype=float)
    m = len(p)
    ordem = np.argsort(p)
    ajustado = np.empty(m)
    acumulado = 0.0
    for posto, i in enumerate(ordem):
        acumulado = max(acumulado, min(1.0, (m - posto) * p[i]))
        ajustado[i] = acumulado
    return ajustado.tolist()


def _aplicar_holm(comparacoes, campo):
    """Acrescenta p_holm e o veredito a alfa=0,05 ao bloco `campo` de cada comparação."""
    nomes = [n for n in comparacoes if campo in comparacoes[n]]
    ajustados = holm([comparacoes[n][campo]['p_bootstrap_corrigido'] for n in nomes])
    for nome, p_adj in zip(nomes, ajustados):
        comparacoes[nome][campo]['p_holm'] = float(p_adj)
        comparacoes[nome][campo]['significativo_holm'] = bool(p_adj < ALFA)
        comparacoes[nome][campo]['tamanho_familia_holm'] = len(nomes)


def mcnemar(y_true, pred_a, pred_b):
    """Teste exato de McNemar sobre acertos e erros discordantes (acurácia)."""
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


def comparar_familia(y_true, ref, outros, W_frase, W_video):
    """Bootstrap por frase e por vídeo (com Holm) e McNemar, ref contra cada outro."""
    comparacoes = {}
    for nome, pred in outros.items():
        comparacoes[nome] = {
            'bootstrap': bootstrap_pareado(W_frase, y_true, ref, pred),
            'mcnemar': mcnemar(y_true, ref, pred),
        }
        if W_video is not None:
            comparacoes[nome]['bootstrap_por_video'] = bootstrap_pareado(W_video, y_true, ref, pred)
    _aplicar_holm(comparacoes, 'bootstrap')
    if W_video is not None:
        _aplicar_holm(comparacoes, 'bootstrap_por_video')
    return comparacoes


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Comparação pareada contra o BERTimbau (M3d)')
    ap.add_argument('--resultados', default=None,
                    help='pasta de onde ler modelo_final/ e baselines_ajustados_* '
                         '(padrão: experimento.dir_resultados())')
    ap.add_argument('--saida', default=None,
                    help='pasta do JSON de saída (padrão: experimento.dir_resultados())')
    ap.add_argument('--sem-sonda', action='store_true',
                    help='não roda a sonda linear (que extrai representações do BERTimbau)')
    ap.add_argument('--sem-sementes', action='store_true',
                    help='não repete a comparação para cada semente do treino final')
    args = ap.parse_args()

    from app.nlp.datasets.prepare_data_toxicity import get_data_for_cv_and_test

    pasta = os.path.abspath(args.resultados) if args.resultados else str(experimento.dir_resultados())
    destino_dir = os.path.abspath(args.saida) if args.saida else str(experimento.dir_resultados())
    os.makedirs(destino_dir, exist_ok=True)

    (X_train, y_train), (X_test, y_test), (df, _, test_idx) = get_data_for_cv_and_test(
        test_size=0.20, return_frame=True)
    y_test = np.asarray(y_test)
    ids_teste = df['id_sentenca'].values[test_idx]
    videos_teste = df['ID'].astype(str).values[test_idx] if 'ID' in df.columns else None
    logger.info(f"Teste: {len(X_test)} frases"
                + (f", {len(set(videos_teste))} vídeos" if videos_teste is not None else ''))

    fontes = {'pasta_resultados': pasta}
    bert = predicoes_bertimbau_csv(pasta, ids_teste, y_test)
    if bert is not None:
        pred_ref = bert['por_semente'][bert['semente_referencia']]
        fontes.update({k: v for k, v in bert.items() if k != 'por_semente'})
        fontes['predicoes_bertimbau'] = 'CSV do treino final (semente de referência)'
    else:
        pred_ref, nome_modelo = predicoes_bertimbau_modelo(list(X_test))
        fontes.update({'predicoes_bertimbau': 'predição do modelo de referência',
                       'modelo_referencia': nome_modelo})

    outros = {}
    if not args.sem_sonda:
        logger.info("Sonda linear...")
        outros[SONDA] = predicoes_sonda(X_train, y_train, list(X_test))
    logger.info("Baselines TF-IDF fixas...")
    outros.update(predicoes_baselines(X_train, y_train, list(X_test)))
    ajustadas, arq_ajust = predicoes_baselines_ajustadas(pasta, ids_teste, y_test)
    fontes['predicoes_baselines_ajustadas'] = arq_ajust
    if ajustadas:
        logger.info(f"Baselines ajustadas: {', '.join(ajustadas)} ({arq_ajust})")
        outros.update(ajustadas)

    W_frase = pesos_bootstrap(len(y_test))
    W_video, n_videos = (pesos_bootstrap_por_grupo(videos_teste) if videos_teste is not None
                         else (None, None))

    predicoes = {REFERENCIA: pred_ref, **outros}
    f1s = {n: float(f1_score(y_test, p, average='macro', zero_division=0))
           for n, p in predicoes.items()}
    comparacoes = comparar_familia(y_test, pred_ref, outros, W_frase, W_video)

    print("\n" + "=" * 96)
    print("F1 MACRO NO TESTE RETIDO")
    print("=" * 96)
    for nome, f1 in sorted(f1s.items(), key=lambda kv: -kv[1]):
        print(f"  {nome:<36} {f1:.4f}")

    print("\n" + "=" * 96)
    print(f"DIFERENÇA DE F1 MACRO CONTRA {REFERENCIA} — bootstrap pareado, IC 95%, Holm "
          f"({len(comparacoes)} comparações)")
    print("=" * 96)
    print(f"  {'Contra':<34} {'Δ F1':>8} {'IC 95% (frases)':>18} {'p Holm':>8} "
          f"{'IC 95% (vídeos)':>18} {'p Holm':>8} {'McNemar':>8}")
    print("-" * 96)
    for nome, c in comparacoes.items():
        bs, bv = c['bootstrap'], c.get('bootstrap_por_video')
        ic = f"[{bs['ic95_baixo']:+.3f}, {bs['ic95_alto']:+.3f}]"
        icv = f"[{bv['ic95_baixo']:+.3f}, {bv['ic95_alto']:+.3f}]" if bv else '—'
        pv = f"{bv['p_holm']:.4f}" if bv else '—'
        print(f"  {nome:<34} {bs['diferenca_observada']:>+8.4f} {ic:>18} {bs['p_holm']:>8.4f} "
              f"{icv:>18} {pv:>8} {c['mcnemar']['p_valor']:>8.4f}")
    print("=" * 96)
    print("McNemar compara acurácia (frases em que só um acerta), não F1 macro: complementar.")

    # --- robustez à semente do BERTimbau ------------------------------------
    por_semente = None
    if bert is not None and not args.sem_sementes and len(bert['por_semente']) > 1:
        por_semente = {'por_semente': {}, 'resumo': {}}
        for semente, pred in bert['por_semente'].items():
            comp = comparar_familia(y_test, pred, outros, W_frase, W_video)
            por_semente['por_semente'][str(semente)] = {
                'f1_macro_bertimbau': float(f1_score(y_test, pred, average='macro', zero_division=0)),
                'comparacoes': comp,
            }
        for nome in outros:
            blocos = [v['comparacoes'][nome] for v in por_semente['por_semente'].values()]
            dif = [b['bootstrap']['diferenca_observada'] for b in blocos]
            resumo = {
                'n_sementes': len(blocos),
                'diferenca_min': float(np.min(dif)),
                'diferenca_max': float(np.max(dif)),
                'diferenca_media': float(np.mean(dif)),
                'n_sementes_significativo_holm_frases': int(sum(b['bootstrap']['significativo_holm']
                                                                for b in blocos)),
                'p_holm_frases_max': float(max(b['bootstrap']['p_holm'] for b in blocos)),
            }
            if W_video is not None:
                resumo['n_sementes_significativo_holm_videos'] = int(
                    sum(b['bootstrap_por_video']['significativo_holm'] for b in blocos))
                resumo['p_holm_videos_max'] = float(max(b['bootstrap_por_video']['p_holm']
                                                        for b in blocos))
            por_semente['resumo'][nome] = resumo

        print(f"\nRobustez à semente ({len(bert['por_semente'])} sementes do treino final): "
              "Δ F1 mín–máx | sementes com diferença detectada (Holm, frases / vídeos)")
        for nome, r in por_semente['resumo'].items():
            sv = r.get('n_sementes_significativo_holm_videos', '—')
            print(f"  {nome:<34} {r['diferenca_min']:+.4f} a {r['diferenca_max']:+.4f} | "
                  f"{r['n_sementes_significativo_holm_frases']}/{r['n_sementes']} / {sv}/{r['n_sementes']}")

    destino = os.path.join(destino_dir, f'comparacao_pareada_{datetime.now():%Y%m%d_%H%M%S}.json')
    saida = {
        'f1_macro_teste': f1s,
        'comparacoes_vs_' + REFERENCIA: comparacoes,
        'n_teste': int(len(y_test)),
        'n_videos_teste': n_videos,
        'n_bootstrap': N_BOOTSTRAP,
        'semente_bootstrap': RANDOM_STATE,
        'alfa': ALFA,
        'metodo': {
            'bootstrap': ('bootstrap pareado por frase: reamostra as frases do teste com '
                          'reposição; mesmas reamostragens para todas as comparações'),
            'bootstrap_por_video': ('bootstrap pareado por vídeo (cluster): reamostra os vídeos '
                                    'do teste com reposição, cada um com todas as suas frases'),
            'p_valor': METODO_P,
            'holm': ('correção de Holm sobre p_bootstrap_corrigido, família = todas as '
                     'comparações contra o BERTimbau deste arquivo (feita em separado para o '
                     'bootstrap por frase e por vídeo, e para cada semente)'),
            'mcnemar': ('complementar: teste exato (binomial) sobre as frases em que só um dos '
                        'dois modelos acerta; compara acurácia pareada, não F1 macro. so_a = só '
                        'o BERTimbau acerta; so_b = só o outro acerta'),
        },
        'fontes': fontes,
        'experimento': experimento.descricao(),
    }
    if por_semente is not None:
        saida['robustez_sementes_bertimbau'] = por_semente
    with open(destino, 'w', encoding='utf-8') as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)
    print(f"\nSalvo em: {destino}")


if __name__ == '__main__':
    main()
