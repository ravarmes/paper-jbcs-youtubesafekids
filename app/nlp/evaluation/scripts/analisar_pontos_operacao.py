"""
Pontos de operação do detector no conjunto de teste retido.

As Tabelas 6 e 7 do manuscrito (efeito do limiar tau sobre a probabilidade da classe
Severa, e a comparação de pontos de operação contra o modelo generativo) eram os únicos
números do artigo sem script que os produzisse. Este script fecha essa lacuna.

O QUE CALCULA
-------------
  1. Para cada limiar tau, sinalizando como severamente tóxica toda frase cuja
     probabilidade da classe Severa passe de tau:
       - recall da classe Severa
       - quantas Severas ficam sem sinalização
       - falso alarme sobre a classe Nenhuma (fração de Nenhuma acima de tau)
  2. Os pontos de operação comparáveis: BERTimbau por argmax, BERTimbau com o
     melhor tau e o modelo generativo (lido do llm_baselines_toxicity_*.json mais
     recente), cada um com DUAS colunas de falso alarme -- ver a nota em
     pontos_de_operacao() sobre por que uma só não serve.
  3. As frases da classe Severa preditas como Nenhuma, com o score de segurança de
     cada uma, contra a mediana das Nenhuma corretamente classificadas — que é o
     teste de se o score contínuo recupera o que o argmax perde.

Uso:
    python -m app.nlp.evaluation.scripts.analisar_pontos_operacao
    python -m app.nlp.evaluation.scripts.analisar_pontos_operacao --idioma en
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp.models.bertimbau_toxicity import BertimbauToxicity  # noqa: E402

CLASSES = ['Nenhuma', 'Leve', 'Severa']
PESOS = [0.0, 0.5, 1.0]
TAUS = [0.50, 0.20, 0.15, 0.10]

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'
)
# project_root aponta para src/; as tabelas do manuscrito ficam um nivel acima.
LATEX_TABELAS = os.path.join(os.path.dirname(project_root), 'latex', 'tabelas')


def modelo_mais_recente() -> str:
    trained = Path(project_root) / 'src' / 'app' / 'nlp' / 'models' / 'trained'
    if not trained.is_dir():
        trained = Path(project_root) / 'app' / 'nlp' / 'models' / 'trained'
    candidatos = [p for p in trained.glob('TOX_*') if (p / 'config.json').exists()]
    if not candidatos:
        raise FileNotFoundError(f"Nenhum modelo TOX_* treinado em {trained}")
    return str(max(candidatos, key=lambda p: p.stat().st_mtime))


def probabilidades_do_teste(modelo, batch_size=64):
    import torch
    from app.nlp.datasets.prepare_data_toxicity import get_data_for_cv_and_test

    _, (textos, rotulos) = get_data_for_cv_and_test(test_size=0.20)
    saidas = []
    modelo.model.eval()
    with torch.no_grad():
        for i in range(0, len(textos), batch_size):
            entradas = modelo.tokenizer(
                textos[i:i + batch_size], padding='max_length', truncation=True,
                max_length=modelo.max_length, return_tensors='pt'
            )
            entradas = {k: v.to(modelo.device) for k, v in entradas.items()}
            probs = torch.softmax(modelo.model(**entradas).logits, dim=-1).cpu().numpy()
            saidas.append(probs)
    return textos, np.asarray(rotulos), np.vstack(saidas)


def curva_de_limiar(y, probs):
    """Efeito de tau sobre a probabilidade da classe Severa (Tabela 6)."""
    p_severa = probs[:, 2]
    eh_severa = y == 2
    eh_nenhuma = y == 0
    linhas = []
    for tau in TAUS:
        sinalizada = p_severa > tau
        recall = float(sinalizada[eh_severa].mean())
        linhas.append({
            'tau': tau,
            'recall_severa': recall,
            'severas_nao_sinalizadas': int((~sinalizada[eh_severa]).sum()),
            'falso_alarme_nenhuma': float(sinalizada[eh_nenhuma].mean()),
        })
    return linhas


def pontos_de_operacao(y, probs, llm):
    """Tabela 7: argmax, melhor tau e o modelo generativo lado a lado.

    DUAS colunas de falso alarme, e nao uma. As linhas nao sinalizam pela mesma
    regra -- o argmax e o modelo generativo entregam uma classe, o limiar entrega
    P(Severa) > tau -- e colapsar as duas sob um unico rotulo "apropriado
    sinalizado" fazia o limiar parecer dominar o argmax em recall E em falso
    alarme ao mesmo tempo (97,5% contra 95,1% e 7,1% contra 7,9%). Nenhuma regra
    monotona permite isso: como {argmax = Severa} esta contido em
    {P(Severa) > 0,1}, afrouxar o limiar so pode aumentar o conjunto sinalizado.
    O que produzia o efeito era a troca de regra no meio da coluna.

    Cada coluna compara agora o que e comparavel, e as duas sao monotonas:
      - fa_severa:  a frase apropriada e marcada como SEVERAMENTE toxica
      - fa_toxico:  a frase apropriada e marcada como toxica em QUALQUER grau
    """
    pred = probs.argmax(1)
    p_severa = probs[:, 2]
    eh_severa = y == 2
    eh_nenhuma = y == 0

    pontos = [{
        'modelo': 'BERTimbau (argmax)',
        'recall_severa': float((pred[eh_severa] == 2).mean()),
        'severas_perdidas': int((pred[eh_severa] != 2).sum()),
        'fa_severa': float((pred[eh_nenhuma] == 2).mean()),
        'fa_toxico': float((pred[eh_nenhuma] != 0).mean()),
    }]

    tau = 0.10
    sinalizada = p_severa > tau
    # Sob a regra "qualquer grau", baixar o limiar acrescenta as Severas ao que o
    # argmax ja marcava como Leve -- dai a uniao, e nao so o limiar.
    pontos.append({
        'modelo': f'BERTimbau (tau = {tau})',
        'recall_severa': float(sinalizada[eh_severa].mean()),
        'severas_perdidas': int((~sinalizada[eh_severa]).sum()),
        'fa_severa': float(sinalizada[eh_nenhuma].mean()),
        'fa_toxico': float(((pred != 0) | sinalizada)[eh_nenhuma].mean()),
    })

    if llm:
        # A matriz do LLM vem na mesma ordem de classes; a linha Severa da o recall,
        # e a linha Nenhuma, quanto do conteudo apropriado ele marca como toxico.
        m = np.asarray(llm['confusion_matrix'])
        pontos.append({
            'modelo': llm['nome'],
            'recall_severa': float(m[2, 2] / m[2].sum()),
            'severas_perdidas': int(m[2].sum() - m[2, 2]),
            'fa_severa': float(m[0, 2] / m[0].sum()),
            'fa_toxico': float((m[0].sum() - m[0, 0]) / m[0].sum()),
            'segundos_por_frase': llm.get('segundos_por_frase'),
        })
    return pontos


def severas_perdidas(textos, y, probs):
    """As Severas que o argmax deixa passar como Nenhuma, e seus scores."""
    pred = probs.argmax(1)
    scores = 1.0 - probs @ np.asarray(PESOS)
    perdidas = [
        {'texto': textos[i], 'score': float(scores[i]),
         'p_severa': float(probs[i, 2])}
        for i in range(len(y)) if y[i] == 2 and pred[i] == 0
    ]
    nenhuma_ok = scores[(y == 0) & (pred == 0)]
    return {
        'n': len(perdidas),
        'frases': sorted(perdidas, key=lambda d: d['score']),
        'score_mediano_perdidas': float(np.median([p['score'] for p in perdidas]))
        if perdidas else None,
        'score_mediano_nenhuma_corretas': float(np.median(nenhuma_ok)),
    }


def carrega_llm():
    fs = sorted(glob.glob(os.path.join(RESULTS_DIR, 'llm_baselines_toxicity_*.json')))
    if not fs:
        return None
    with open(fs[-1], encoding='utf-8') as f:
        d = json.load(f)
    nome, r = next(iter(d['results'].items()))
    n = r.get('n_amostras') or sum(sum(l) for l in r['confusion_matrix'])
    return {
        'nome': nome, 'confusion_matrix': r['confusion_matrix'],
        'segundos_por_frase': round(r['segundos'] / n, 2) if r.get('segundos') else None,
        'fonte': os.path.basename(fs[-1]),
    }


def tex_limiar(linhas, idioma, argmax_coincide):
    # A anotacao "(argmax)" na linha 0,50 so entra se o argmax de fato produzir os
    # mesmos numeros. Nao e identidade: com tres classes o argmax pode escolher
    # Severa com P(Severa) < 0,5. Coincide neste conjunto de teste, e o script
    # confere em vez de supor -- se um retreino separar os dois, a anotacao some.
    en = idioma == 'en'
    ident = (" In practice, \\emph{argmax} coincides with $\\tau = 0.5$ on this test set."
             if en else " Na prática, o \\emph{argmax} coincide com $\\tau = 0{,}5$ neste "
             "conjunto de teste.") if argmax_coincide else ""
    cap = (("Effect of the threshold $\\tau$ on the probability of the Severe class. Every "
            "row flags by the same rule, $P(\\text{Severe}) > \\tau$, so relaxing $\\tau$ "
            "can only enlarge the flagged set." + ident) if en else
           ("Efeito do limiar $\\tau$ sobre a probabilidade da classe Severa. Todas as "
            "linhas sinalizam pela mesma regra, $P(\\text{Severa}) > \\tau$, de modo que "
            "afrouxar $\\tau$ só pode aumentar o conjunto sinalizado." + ident))
    cab = ("$\\tau$ & Severe recall & Severe not flagged & False alarm on None" if en else
           "$\\tau$ & \\emph{Recall} Severa & Severas não sinalizadas & Falso alarme em Nenhuma")
    dec = (lambda x: f"{x:.1f}\\%") if en else (lambda x: f"{x:.1f}\\%".replace('.', '{,}'))
    melhor = max(linhas, key=lambda l: l['recall_severa'])
    out = ["% Gerado por app/nlp/evaluation/scripts/analisar_pontos_operacao.py",
           "% NAO EDITAR A MAO: reexecute o script apos retreinar.",
           "\\begin{table*}[!t]", "\\centering", f"\\caption{{{cap}}}",
           "\\label{tab:limiar}", "\\begin{tabular}{cccc}", "\\toprule",
           cab + " \\\\", "\\midrule"]
    for l in linhas:
        rot = f"{l['tau']:.2f}" if en else f"{l['tau']:.2f}".replace('.', '{,}')
        if l['tau'] == 0.50:
            rot += " (\\emph{argmax})"
        cells = [rot, dec(100 * l['recall_severa']), str(l['severas_nao_sinalizadas']),
                 dec(100 * l['falso_alarme_nenhuma'])]
        if l is melhor:
            cells = [f"\\textbf{{{c}}}" for c in cells]
        out.append(" & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(out)


def tex_pontos(pontos, idioma):
    en = idioma == 'en'
    cap = ("Operating points on the test set. The expensive error is letting severely "
           "toxic content through; the cheap one is flagging appropriate content, which "
           "merely demotes it in the ranking. False alarm is reported under both "
           "flagging rules, because the rows do not flag alike: \\emph{argmax} and the "
           "generative model return a class, whereas the threshold is applied to "
           "$P(\\text{Severe})$. Reading a single rule down the column would compare "
           "different policies; under either rule taken on its own, relaxing the "
           "threshold costs false alarms, as it must." if en else
           "Pontos de operação no conjunto de teste. O erro caro é deixar passar conteúdo "
           "severamente tóxico; o barato é sinalizar conteúdo apropriado, que apenas o "
           "rebaixa no ranqueamento. O falso alarme vai sob as duas regras de "
           "sinalização, porque as linhas não sinalizam do mesmo jeito: o \\emph{argmax} "
           "e o modelo generativo devolvem uma classe, enquanto o limiar se aplica a "
           "$P(\\text{Severa})$. Ler uma regra só ao longo da coluna compararia "
           "políticas diferentes; sob qualquer uma delas isolada, afrouxar o limiar "
           "custa falso alarme, como tem de custar.")
    cab = ("Model & Severe recall & Severe missed & \\multicolumn{2}{c}{Appropriate flagged} \\\\\n"
           "\\cmidrule(lr){4-5}\n"
           " & & & as Severe & as toxic (any) " if en else
           "Modelo & \\emph{Recall} Severa & Severas perdidas & \\multicolumn{2}{c}{Apropriado sinalizado} \\\\\n"
           "\\cmidrule(lr){4-5}\n"
           " & & & como Severa & como tóxico (qualquer) ")
    dec = (lambda x: f"{x:.1f}\\%") if en else (lambda x: f"{x:.1f}\\%".replace('.', '{,}'))
    out = ["% Gerado por app/nlp/evaluation/scripts/analisar_pontos_operacao.py",
           "% NAO EDITAR A MAO: reexecute o script apos retreinar.",
           "\\begin{table*}[!t]", "\\centering", f"\\caption{{{cap}}}",
           "\\label{tab:pontos-operacao}", "\\begin{tabular}{lcccc}", "\\toprule",
           cab + " \\\\", "\\midrule"]
    for i, p in enumerate(pontos):
        cells = [p['modelo'].replace('tau', '$\\tau$'), dec(100 * p['recall_severa']),
                 str(p['severas_perdidas']), dec(100 * p['fa_severa']),
                 dec(100 * p['fa_toxico'])]
        if i == 1:
            cells = [f"\\textbf{{{c}}}" for c in cells]
        out.append(" & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description='Pontos de operação do detector')
    ap.add_argument('--idioma', choices=['pt', 'en'], default='pt')
    args = ap.parse_args()

    caminho = modelo_mais_recente()
    print(f"Modelo: {os.path.basename(caminho)}")
    modelo = BertimbauToxicity(model_path=caminho)

    textos, y, probs = probabilidades_do_teste(modelo)
    linhas = curva_de_limiar(y, probs)
    llm = carrega_llm()
    pontos = pontos_de_operacao(y, probs, llm)
    perdidas = severas_perdidas(textos, y, probs)

    print("\n--- efeito do limiar sobre a classe Severa ---")
    for l in linhas:
        print(f"  tau={l['tau']:.2f}  recall={100*l['recall_severa']:.1f}%  "
              f"nao sinalizadas={l['severas_nao_sinalizadas']}  "
              f"falso alarme={100*l['falso_alarme_nenhuma']:.1f}%")

    print("\n--- pontos de operacao ---")
    for p in pontos:
        extra = (f"  ({p['segundos_por_frase']} s/frase)"
                 if p.get('segundos_por_frase') else "")
        print(f"  {p['modelo']:<24} recall={100*p['recall_severa']:.1f}%  "
              f"perdidas={p['severas_perdidas']}  "
              f"FA(Severa)={100*p['fa_severa']:.1f}%  "
              f"FA(qualquer)={100*p['fa_toxico']:.1f}%{extra}")

    # A linha 0,50 da curva so pode ser rotulada "(argmax)" se bater com o argmax.
    l50 = next(l for l in linhas if l['tau'] == 0.50)
    argmax_coincide = (
        abs(l50['recall_severa'] - pontos[0]['recall_severa']) < 1e-9
        and l50['severas_nao_sinalizadas'] == pontos[0]['severas_perdidas']
        and abs(l50['falso_alarme_nenhuma'] - pontos[0]['fa_severa']) < 1e-9
    )
    print(f"\n  argmax coincide com tau=0,50: {argmax_coincide}")

    print(f"\n--- Severas preditas como Nenhuma: {perdidas['n']} ---")
    for p in perdidas['frases']:
        print(f"  score={p['score']:.3f}  p(Severa)={p['p_severa']:.3f}  {p['texto'][:70]}")
    print(f"  mediana do score nas perdidas: {perdidas['score_mediano_perdidas']}")
    print(f"  mediana do score nas Nenhuma corretas: "
          f"{perdidas['score_mediano_nenhuma_corretas']:.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    destino = os.path.join(RESULTS_DIR, f'pontos_operacao_{stamp}.json')
    with open(destino, 'w', encoding='utf-8') as f:
        json.dump({'modelo': os.path.basename(caminho), 'n_teste': int(len(y)),
                   'curva_limiar': linhas, 'pontos_operacao': pontos,
                   'argmax_coincide_tau_050': bool(argmax_coincide),
                   'severas_perdidas': perdidas, 'llm': llm},
                  f, indent=2, ensure_ascii=False)
    print(f"\nSalvo em: {destino}")

    os.makedirs(LATEX_TABELAS, exist_ok=True)
    for nome, conteudo in (('tab-limiar.tex', tex_limiar(linhas, args.idioma, argmax_coincide)),
                           ('tab-pontos-operacao.tex', tex_pontos(pontos, args.idioma))):
        alvo = os.path.join(LATEX_TABELAS, nome)
        with open(alvo, 'w', encoding='utf-8') as f:
            f.write(conteudo)
        print(f"Tabela LaTeX: {alvo}")


if __name__ == '__main__':
    main()
