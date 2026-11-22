"""
Configuração da rodada experimental: partição e rotulagem.

Duas escolhas atravessam o pipeline inteiro e precisam ser as mesmas em todos os
scripts de uma rodada. Ambas são lidas de variáveis de ambiente, de modo que
nenhum script precise de argumento novo:

YSK_PARTICAO  'video' (padrão) | 'frase'
    'video'  -- todas as frases de um vídeo ficam do mesmo lado em toda divisão:
                teste retido, folds da validação cruzada e dev interno usado no
                early stopping. É a unidade em que o serviço opera (o vídeo, não a
                frase) e passou a ser o protocolo principal na revisão do JBCS.
    'frase'  -- a divisão da versão submetida: estratificada por frase, sem
                agrupar por vídeo. Mantida para a comparação secundária.

YSK_ROTULAGEM 'v2' (padrão) | 'v1'
    'v2'       -- os mesmos rótulos do v1, mas os experimentos deixam de fora as
                  frases dos grupos de quase-duplicatas (coluna `quase_duplicata`
                  da partição; regra em app/nlp/datasets/quase_duplicatas.py).
                  Elas não entram no treino, nos folds de validação, no dev
                  interno nem no teste: carregar_particao_video() as devolve com
                  particao = 'excluida'. Os vídeos de cada lado são os mesmos do v1.
    'v1'       -- o corpus v1 (revisão do JBCS): 1.811 frases de 206 vídeos, numa
                  única rodada de anotação, com rótulo final = moda das três
                  anotações (desempate pelo P1). O v1 tirou do corpus submetido as
                  270 frases cujo link não leva a vídeo e acrescentou 575 frases
                  anotadas pelos mesmos critérios.
    As rotulagens do corpus submetido ('revisada' e 'original', 1.506 frases)
    não são mais aceitas: o corpus de data/ é o v1. Seus resultados continuam em
    evaluation/results/video-revisada/ e video-original/.

Cada combinação grava resultados e modelos em subpastas próprias
(evaluation/results/<tag>/ e models/trained/<tag>/), para que a busca pelo
"modelo mais recente" de um script nunca pegue o artefato de outra rodada. A
combinação 'frase-revisada' é a da versão submetida: seus artefatos ficaram na
raiz dessas pastas e continuam lá.

As partições por vídeo são calculadas UMA vez (app/nlp/datasets/gerar_particoes.py)
e gravadas em data/particoes/. Ler a partição do disco, em vez de recalculá-la,
garante que todos os scripts de uma rodada usem exatamente os mesmos vídeos de
treino, de validação e de teste.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

PARTICOES_VALIDAS = ('video', 'frase')
ROTULAGENS_VALIDAS = ('v1', 'v2')

# Raiz de src/
SRC_ROOT = Path(__file__).resolve().parents[2]
NLP_ROOT = SRC_ROOT / 'app' / 'nlp'
DATA_DIR = SRC_ROOT / 'data'
PARTICOES_DIR = DATA_DIR / 'particoes'
ARQ_PARTICAO_VIDEO = PARTICOES_DIR / 'particao_video.csv'
ARQ_PARTICAO_FRASE = PARTICOES_DIR / 'particao_frase.csv'

# Configuração da versão submetida: artefatos na raiz de results/ e trained/
TAG_LEGADO = 'frase-revisada'

# Folds pré-calculados no arquivo de partição por vídeo (estágio 1 da busca usa
# 2 folds; os demais, 5).
FOLDS_DISPONIVEIS = (2, 5)


def particao() -> str:
    valor = os.getenv('YSK_PARTICAO', 'video').strip().lower()
    if valor not in PARTICOES_VALIDAS:
        raise ValueError(f"YSK_PARTICAO={valor!r}; use um de {PARTICOES_VALIDAS}")
    return valor


def rotulagem() -> str:
    valor = os.getenv('YSK_ROTULAGEM', 'v2').strip().lower()
    if valor not in ROTULAGENS_VALIDAS:
        raise ValueError(f"YSK_ROTULAGEM={valor!r}; use um de {ROTULAGENS_VALIDAS}")
    return valor


def tag() -> str:
    """Identificador da rodada, ex.: 'video-v1'."""
    return f"{particao()}-{rotulagem()}"


def descricao() -> dict:
    """Metadados da rodada, para gravar junto de cada resultado."""
    return {
        'particao': particao(),
        'rotulagem': rotulagem(),
        'tag': tag(),
        'arquivo_particao': (str(ARQ_PARTICAO_VIDEO.relative_to(SRC_ROOT))
                             if particao() == 'video' else
                             str(ARQ_PARTICAO_FRASE.relative_to(SRC_ROOT))),
    }


def dir_resultados(*sub) -> Path:
    """Pasta de resultados da rodada (criada se não existir)."""
    base = NLP_ROOT / 'evaluation' / 'results'
    if tag() != TAG_LEGADO:
        base = base / tag()
    destino = base.joinpath(*sub) if sub else base
    destino.mkdir(parents=True, exist_ok=True)
    return destino


def dir_modelos() -> Path:
    """Pasta dos modelos treinados da rodada (criada se não existir)."""
    base = NLP_ROOT / 'models' / 'trained'
    if tag() != TAG_LEGADO:
        base = base / tag()
    base.mkdir(parents=True, exist_ok=True)
    return base


ARTIGO_ROOT = SRC_ROOT.parent


def dir_latex(sub: str) -> Path:
    """
    Destino das tabelas e figuras que os scripts geram para o manuscrito.

    Só escreve em latex/<sub> com YSK_ESCREVER_LATEX=1. Sem isso, grava em
    results/<tag>/latex/<sub>: rodar uma configuração secundária (ou uma rodada
    ainda em conferência) não sobrescreve as tabelas do artigo. Os números entram
    no manuscrito num passo deliberado, com a configuração escolhida.
    """
    if os.getenv('YSK_ESCREVER_LATEX', '').strip() == '1':
        destino = ARTIGO_ROOT / 'latex' / sub
    else:
        destino = dir_resultados('latex', sub)
    destino.mkdir(parents=True, exist_ok=True)
    return destino


ARQ_MODELO_REFERENCIA = 'modelo_referencia.txt'


def registrar_modelo_referencia(caminho) -> None:
    """Marca qual modelo final é o de referência para as análises a jusante."""
    (dir_modelos() / ARQ_MODELO_REFERENCIA).write_text(Path(caminho).name, encoding='utf-8')


def caminho_modelo_referencia() -> Path:
    """
    Modelo de referência da rodada.

    Usa o ponteiro gravado pelo treino final (semente de referência); na falta
    dele, o TOX_* mais recente com pesos salvos -- comportamento da versão
    submetida, em que só havia um modelo final.
    """
    pasta = dir_modelos()
    ponteiro = pasta / ARQ_MODELO_REFERENCIA
    if ponteiro.exists():
        caminho = pasta / ponteiro.read_text(encoding='utf-8').strip()
        if _tem_pesos(caminho):
            return caminho
    candidatos = [p for p in pasta.glob('TOX_*') if _tem_pesos(p)]
    if not candidatos:
        raise FileNotFoundError(f"Nenhum modelo TOX_* com pesos em {pasta}")
    return max(candidatos, key=lambda p: p.stat().st_mtime)


def _tem_pesos(pasta: Path) -> bool:
    """Modelo utilizável: config e pesos presentes (sementes descartadas só têm config)."""
    return ((pasta / 'config.json').exists() and
            ((pasta / 'model.safetensors').exists() or (pasta / 'pytorch_model.bin').exists()))


# ---------------------------------------------------------------------------
# Partição por vídeo
# ---------------------------------------------------------------------------

def exclui_quase_duplicatas() -> bool:
    """Na rodada v2, os grupos de quase-duplicatas ficam fora dos experimentos."""
    return rotulagem() == 'v2'


def _ler_particao(arquivo: Path) -> pd.DataFrame:
    if not arquivo.exists():
        raise FileNotFoundError(
            f"Partição não encontrada em {arquivo}. "
            "Gere com: python -m app.nlp.datasets.gerar_particoes"
        )
    part = pd.read_csv(arquivo, sep=';', encoding='utf-8-sig', dtype={'video_id': str})
    if exclui_quase_duplicatas():
        if 'quase_duplicata' not in part.columns:
            raise ValueError(f"{arquivo.name} sem a coluna quase_duplicata; "
                             "rode python -m app.nlp.datasets.gerar_particoes")
        part.loc[part['quase_duplicata'].astype(int) == 1, 'particao'] = 'excluida'
    return part


def carregar_particao_video() -> pd.DataFrame:
    """Partição por vídeo; na v2, as quase-duplicatas vêm com particao = 'excluida'."""
    return _ler_particao(ARQ_PARTICAO_VIDEO)


def ids_excluidos() -> set:
    """id_sentenca das frases fora dos experimentos (vazio na v1)."""
    if not exclui_quase_duplicatas():
        return set()
    part = carregar_particao_video()
    return set(part.loc[part['particao'] == 'excluida', 'id_sentenca'].astype(int))


def folds_por_video(grupos, n_splits: int):
    """
    Índices (treino, validação) de cada fold, lidos da partição gravada.

    `grupos` são os video_id das frases do conjunto de desenvolvimento, na ordem
    em que aparecem nos arrays de treino. Cada vídeo pertence a um único fold, o
    mesmo para qualquer rotulagem.
    """
    if n_splits not in FOLDS_DISPONIVEIS:
        raise ValueError(f"Partição por vídeo só tem folds pré-calculados para "
                         f"k em {FOLDS_DISPONIVEIS}; pedido k={n_splits}")
    part = carregar_particao_video()
    coluna = f'fold_k{n_splits}'
    mapa = (part[part['particao'] == 'treino']
            .drop_duplicates('video_id')
            .set_index('video_id')[coluna].astype(int).to_dict())
    grupos = np.asarray(grupos).astype(str)
    faltando = sorted(set(grupos) - set(mapa))
    if faltando:
        raise ValueError(f"{len(faltando)} vídeos sem fold na partição gravada "
                         f"(ex.: {faltando[:3]}); a partição está desatualizada?")
    fold_de = np.array([mapa[g] for g in grupos])
    for f in range(1, n_splits + 1):
        val = np.where(fold_de == f)[0]
        tr = np.where(fold_de != f)[0]
        yield tr, val
