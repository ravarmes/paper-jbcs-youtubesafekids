"""
Roda o pipeline de toxicidade inteiro para uma configuração (partição × rotulagem:
'v2', o padrão, usa os rótulos do corpus v1 sem os grupos de quase-duplicatas; 'v1'
reproduz a rodada com eles).

Cada etapa é um processo separado, com as variáveis YSK_PARTICAO e YSK_ROTULAGEM
definidas (ver app/nlp/experimento.py). A saída de cada etapa vai para
app/nlp/evaluation/results/<tag>/logs/<etapa>.log, e um resumo com status e
duração de cada etapa para results/<tag>/rodada_<timestamp>.json.

Se uma etapa essencial falhar (busca ou treino final), as seguintes, que dependem
do modelo, são puladas; as demais falhas são registradas e a rodada continua.

Nada é escrito em latex/: as tabelas e figuras vão para results/<tag>/latex/
(o manuscrito só é atualizado num passo deliberado, com YSK_ESCREVER_LATEX=1).

Uso (de dentro de src/):
    python scripts/rodar_rodada_toxicidade.py --particao video          # v2 (padrão)
    python scripts/rodar_rodada_toxicidade.py --particao video --rotulagem v1
    python scripts/rodar_rodada_toxicidade.py --particao frase          # comparação secundária
    python scripts/rodar_rodada_toxicidade.py ... --pular llm          # sem Ollama
    python scripts/rodar_rodada_toxicidade.py ... --sem-busca          # reaproveita a busca
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]

# (nome, argumentos do python, essencial?)
ETAPAS = [
    ('busca', ['-m', 'app.nlp.training.train_toxicity_gridsearch', '--mode', 'search'], True),
    ('final', ['-m', 'app.nlp.training.train_toxicity_gridsearch', '--mode', 'final'], True),
    ('avaliacao', ['-m', 'app.nlp.evaluation.evaluate_toxicity'], False),
    ('baselines', ['-m', 'app.nlp.training.train_toxicity_baselines'], False),
    # Pelo caminho do arquivo (sem torch): baselines ajustadas por CV aninhada e a
    # dispersão entre folds, que lê o modelo final e as duas baselines.
    ('baselines_ajustadas', ['app/nlp/training/train_toxicity_baselines_ajustados.py'], False),
    ('dispersao', ['app/nlp/evaluation/scripts/analisar_dispersao.py'], False),
    ('sem_finetuning', ['-m', 'app.nlp.training.train_toxicity_sem_finetuning', '--modo', 'tudo'], False),
    ('llm', ['-m', 'app.nlp.training.train_toxicity_llm_baselines', '--models', 'llama3.1:8b'], False),
    # Detoxify multilíngue, sem ajuste fino; o mapeamento para três níveis usa só o treino.
    ('detoxify', ['app/nlp/evaluation/scripts/avaliar_detoxify.py'], False),
    ('comparacao', ['-m', 'app.nlp.evaluation.scripts.comparar_modelos'], False),
    # Pelo caminho do arquivo (sem torch): F1 do teste por lote de anotação (M2) e época
    # restaurada no treino final (M3c); lê as predições gravadas e o log do treino final.
    ('teste_por_lote', ['app/nlp/evaluation/scripts/analisar_teste_por_lote.py'], False),
    # Pelo caminho do arquivo (sem torch): descrição dos grupos de quase-duplicatas e
    # sensibilidade do F1 do teste às regras vizinhas (parecer B1).
    ('quase_duplicatas', ['app/nlp/evaluation/scripts/analisar_quase_duplicatas.py'], False),
    ('pontos_operacao', ['-m', 'app.nlp.evaluation.scripts.analisar_pontos_operacao', '--idioma', 'en'], False),
    ('simulacao', ['-m', 'app.nlp.evaluation.scripts.simular_amostragem'], False),
    ('estratos', ['-m', 'app.nlp.evaluation.scripts.analisar_estratos',
                  '--csv', 'data/reordenacao_buscas.csv', '--idioma', 'en'], False),
    ('scores_exemplo', ['-m', 'app.nlp.evaluation.scripts.calcular_scores_exemplo'], False),
    ('corpus', ['-m', 'app.nlp.evaluation.scripts.analisar_corpus'], False),
    # Pelo caminho do arquivo: sem importar torch (app/nlp/evaluation/__init__.py o importa).
    ('anotacao', ['app/nlp/evaluation/scripts/analisar_anotacao.py'], False),
    ('calibracao_limiar', ['app/nlp/evaluation/scripts/analisar_calibracao_limiar.py'], False),
    # Lê o JSON da etapa anterior: as frases Severas do teste preditas Nenhuma (m6).
    ('severas_nenhuma', ['app/nlp/evaluation/scripts/gerar_tab_severas_nenhuma.py'], False),
    ('figura_resultado', ['app/nlp/evaluation/scripts/gerar_figura_resultado_app.py'], False),
    # Por último: carrega o LLM e o BERTimbau na GPU, um de cada vez.
    ('latencia', ['-m', 'app.nlp.evaluation.scripts.medir_latencia'], False),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--particao', choices=['video', 'frase'], default='video')
    ap.add_argument('--rotulagem', choices=['v1', 'v2'], default='v2')
    ap.add_argument('--pular', nargs='*', default=[], help='etapas a pular (nomes)')
    ap.add_argument('--sem-busca', action='store_true',
                    help='não refaz a busca; usa a mais recente da rodada')
    args = ap.parse_args()

    env = dict(os.environ, YSK_PARTICAO=args.particao, YSK_ROTULAGEM=args.rotulagem,
               PYTHONIOENCODING='utf-8')
    env.pop('YSK_ESCREVER_LATEX', None)
    tag = f'{args.particao}-{args.rotulagem}'
    pasta = SRC / 'app' / 'nlp' / 'evaluation' / 'results' / tag
    logs = pasta / 'logs'
    logs.mkdir(parents=True, exist_ok=True)

    pular = set(args.pular) | ({'busca'} if args.sem_busca else set())
    registro = {'tag': tag, 'inicio': datetime.now().isoformat(), 'etapas': []}
    arquivo = pasta / f'rodada_{datetime.now():%Y%m%d_%H%M%S}.json'
    bloqueado = False

    for nome, argv, essencial in ETAPAS:
        if nome in pular:
            registro['etapas'].append({'etapa': nome, 'status': 'pulada'})
            continue
        if bloqueado:
            registro['etapas'].append({'etapa': nome, 'status': 'não executada (etapa essencial falhou)'})
            continue
        t0 = time.time()
        print(f'[{tag}] {datetime.now():%H:%M:%S} {nome} ...', flush=True)
        with open(logs / f'{nome}.log', 'w', encoding='utf-8') as log:
            rc = subprocess.call([sys.executable] + argv, cwd=SRC, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
        dur = round(time.time() - t0, 1)
        status = 'ok' if rc == 0 else f'falhou (código {rc})'
        registro['etapas'].append({'etapa': nome, 'status': status, 'segundos': dur})
        print(f'[{tag}]   {status} em {dur / 60:.1f} min', flush=True)
        if rc != 0 and essencial:
            bloqueado = True
        registro['fim'] = datetime.now().isoformat()
        arquivo.write_text(json.dumps(registro, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'[{tag}] resumo em {arquivo}')
    sys.exit(1 if bloqueado else 0)


if __name__ == '__main__':
    main()
