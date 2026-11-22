"""
Grupos de quase-duplicatas do corpus: frases que terminam com a mesma sequência de
palavras que pelo menos outras três frases do corpus.

Regra: as três últimas palavras da frase (minúsculas, sem acento, só letras e dígitos)
são iguais às de 4 ou mais frases do corpus, contando a própria. No corpus v1 são 234
frases, 33 terminações, todas tóxicas, em 154 vídeos de séries diferentes.

Por que isso importa para a avaliação: a partição por vídeo não separa essas frases,
porque o grupo atravessa vídeos. Uma frase do teste pode ter quase-duplicatas no treino,
e o modelo a classifica pela terminação. Na rodada video-v1, o BERTimbau acertou 98,6%
dessas frases no teste, contra 63,7% nas demais frases tóxicas.

Por que este limiar: com 3 palavras e 4 ou mais frases, o grupo não contém nenhuma
frase Nenhuma. Afrouxar para 3 ou mais frases já capta repetições de fala comum
("filho da puta", "parava de subir"). Regras vizinhas, mais estritas ou mais frouxas,
são relatadas como análise de sensibilidade (analisar_quase_duplicatas.py).

Na rodada v2 (YSK_ROTULAGEM=v2, ver app/nlp/experimento.py), essas frases ficam fora
de todos os experimentos: treino, folds de validação, dev interno e teste. Elas
continuam no corpus publicado, marcadas na coluna `quase_duplicata`.
"""
import re
import unicodedata

import numpy as np
import pandas as pd

N_PALAVRAS = 3
MIN_FRASES = 4

# Regras vizinhas para a análise de sensibilidade: (palavras, mínimo de frases)
REGRAS_SENSIBILIDADE = ((3, 3), (3, 5), (2, 4), (4, 4))


def palavras(frase) -> list:
    s = unicodedata.normalize('NFKD', str(frase).lower())
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return re.findall(r'[a-z0-9]+', s)


def terminacao(frase, n: int = N_PALAVRAS) -> str:
    return ' '.join(palavras(frase)[-n:])


def marcar(frases, n: int = N_PALAVRAS, minimo: int = MIN_FRASES) -> np.ndarray:
    """Vetor booleano: a frase pertence a um grupo de quase-duplicatas."""
    fim = pd.Series([terminacao(f, n) for f in frases])
    return (fim.map(fim.value_counts()) >= minimo).to_numpy()


def descrever_regra() -> str:
    return (f'as {N_PALAVRAS} últimas palavras (normalizadas) repetidas em '
            f'{MIN_FRASES} ou mais frases do corpus')
