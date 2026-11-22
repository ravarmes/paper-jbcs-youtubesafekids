# Generative baseline: exact prompt, decoding, parsing and invalid outputs

This note documents the few-shot generative baseline of the paper, implemented in
`app/nlp/training/train_toxicity_llm_baselines.py`. Everything below is read from that
script; if the two ever disagree, the script is right and this note is stale. Each
results file (`app/nlp/evaluation/results/<tag>/llm_baselines_toxicity_<timestamp>.json`)
written since the JBCS revision also stores the prompt, the few-shot examples, the
decoding options, the parser rules, every raw reply and every prediction, so a run can be
audited without this note.

## Model and serving

| Item | Value |
|---|---|
| Model | `llama3.1:8b` (Ollama tag), the only model the pipeline runs (`scripts/rodar_rodada_toxicidade.py`: `--models llama3.1:8b`). The script also accepts other tags (default list `llama3.1:8b gemma2:9b`). |
| Server | [Ollama](https://ollama.com), local, `POST /api/chat`, `stream: false` |
| Host | `OLLAMA_HOST`, default `http://localhost:11434` |
| Requests | one per test sentence, sequential, no batching |
| Model identity | recorded per run in the JSON (`ollama.modelos[<tag>].digest`, from `/api/tags`), together with the Ollama version (`ollama.versao`) and how much of the model sat in GPU memory (`ollama.carregados[].size_vram`, from `/api/ps`) |

**Caveat on the model build.** An Ollama tag is a moving pointer. On the authors' machine
the `llama3.1:8b` weights were written to disk on 2026-11-16 (manifest
`~/.ollama/models/manifests/registry.ollama.ai/library/llama3.1/8b`, sha256
`3fc95d284915704cf140b825abf5150af0d32fde95f55c081c9602d49d4547c0`; model layer
`sha256:667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29`, 4,920,738,944
bytes). The run reported in the submitted manuscript predates that date and did not record a
digest, so the exact build it used cannot be stated. The video-v1 run records it.

The Modelfile of this build sets only the Llama 3.1 stop tokens
(`<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`); it does not set `num_ctx`, so
the context length is the server default. The chat template is the one shipped with the
tag.

## Prompt

The request is a chat with a system message, six few-shot turns and the sentence to
classify:

```
[system]    SYSTEM_PROMPT (below)
[user]      Frase: <example 1>
[assistant] <label of example 1, as "0", "1" or "2">
...         (six examples in total)
[user]      Frase: <sentence to classify>
```

The user template is exactly `Frase: {frase}` (Portuguese for "Sentence:"). The system
prompt, verbatim, with its real line breaks:

```
Você é um sistema de moderação de conteúdo em português brasileiro. Sua tarefa é classificar o grau de toxicidade de uma frase extraída da transcrição de um vídeo, em três níveis:
0 = Nenhuma: texto apropriado, isento de linguagem agressiva.
1 = Leve: ofensividade indireta — críticas dirigidas a objetos, situações ou entidades abstratas, sem alvo pessoal (ex.: "Que porra de ônibus").
2 = Severa: ofensividade direta — insultos, ameaças ou discurso de ódio dirigidos explicitamente a uma pessoa ou grupo (ex.: "Este idiota do Robson fez isso").
Responda APENAS com o número 0, 1 ou 2, sem nenhuma outra palavra.
```

English reading aid (not sent to the model): "You are a content-moderation system for
Brazilian Portuguese. Your task is to classify the toxicity level of a sentence extracted
from a video transcript into three levels: 0 = None: appropriate text, free of aggressive
language. 1 = Mild: indirect offensiveness — criticism aimed at objects, situations or
abstract entities, with no personal target (e.g. "What a fucking bus"). 2 = Severe: direct
offensiveness — insults, threats or hate speech explicitly aimed at a person or group
(e.g. "This idiot Robson did this"). Answer ONLY with the number 0, 1 or 2, with no other
word."

The level definitions are the ones given to the human annotators
(`docs/annotation-guideline.md`).

## Few-shot examples

`montar_fewshot` draws **2 examples per class from the training/development split**; the
test split is never used. The rule is deterministic:

1. a single generator `random.Random(42)`;
2. the training sentences are grouped by class, in corpus order;
3. for classes 0, 1 and 2, in this order, the group is shuffled with that generator and its
   first two sentences are taken;
4. the six examples are shuffled once more with the same generator, which fixes their order
   in the prompt.

The examples therefore depend on the corpus version and the split. For the v1 corpus with
the video-grouped split (run tag `video-v1`), they are, in prompt order:

| # | Sentence (as sent) | Label |
|---|---|---|
| 1 | Cala esse focinho, Roger, seu alienígena miserável e insuportável! | 2 |
| 2 | esse falatório estragou nosso dia | 1 |
| 3 | lanchinhos comprados e agora livros | 0 |
| 4 | agora sabemos por US vai se foder Summer e que se for o governo e que eu me foda também por ter abaixado a guarda o que eu nunca farei de novo | 2 |
| 5 | eu perguntei pro Léo ah o Léo disse que gosta de brincar com as fantasias ótimo vamos Léo | 0 |
| 6 | Gastar todo o dinheiro em apostas foi uma tremenda idiotice. | 1 |

Runs on earlier corpus versions (`video-original`, `video-revisada`, and the submitted
sentence-level split) used different examples; their JSON files predate the `prompt` field
and do not list them. They can be regenerated by running `montar_fewshot` on the
corresponding split.

## Decoding

| Option | Value |
|---|---|
| `temperature` | 0 |
| `num_predict` | 5 |
| everything else (`num_ctx`, `top_k`, `top_p`, `seed`, `repeat_penalty`, ...) | server default |
| output format | free text (no `format` / JSON schema) |

The options live in one constant, `OPCOES_OLLAMA`, which is also what the latency script
(`app/nlp/evaluation/scripts/medir_latencia.py`) sends in its "free" mode.

## Parsing

`parse_label` maps the reply to a class. `classificar_saida` records which rule fired,
without changing the prediction. The rules, in order:

| Category in the JSON | Rule | Prediction |
|---|---|---|
| `valida` | the reply, stripped of surrounding whitespace, is exactly `0`, `1` or `2` (the requested format) | that digit |
| `digito_extraido` | otherwise, the reply contains one of the characters `0`, `1`, `2` | the first of them in the text (note: `10` gives 1) |
| `palavra_chave` | otherwise, it contains a class name, tested in the order `severa`, `leve`, `nenhuma` (case-insensitive) | 2, 1 or 0 |
| `sem_rotulo` | empty reply or nothing recognisable | 0 (None), a conservative fallback |

No sentence is discarded, re-asked or excluded from the metrics: every test sentence gets
exactly one prediction. Network failures are retried up to three times (2, 4 and 6 s
back-off); if all three fail, the model is reported as failed and no results file is
written for it.

## Invalid outputs: what was recorded

- **Submitted manuscript and the `video-original` / `video-revisada` re-runs:** the raw
  replies were not stored, so the number of replies outside the requested format is
  **unknown** for those runs. The stored JSON has only the confusion matrix, the metrics and
  the total wall time (`segundos`).
- **From the video-v1 run on:** `results[<model>].saidas.contagem` gives the count per
  category above, `saidas.n_fora_do_formato` the replies that were not exactly one digit,
  and `saidas.n_fallback_nenhuma` those that fell back to None. `respostas_brutas` and
  `y_pred` keep every reply and prediction in test-set order, and
  `latencia_requisicao_s` the wall time of every request.

## Appendix fragment

The script writes `apendice-llm.tex` to `experimento.dir_latex('apendice')`
(`results/<tag>/latex/apendice/` unless `YSK_ESCREVER_LATEX=1`): the exact messages in a
`verbatim` block inside a `figure*` (lines wrapped at 88 characters), followed by the
decoding, few-shot, parsing and invalid-output facts. It is written at the end of every
run and can be regenerated without calling Ollama:

```bash
python -m app.nlp.training.train_toxicity_llm_baselines --so-apendice --models llama3.1:8b
```

With `--so-apendice`, the invalid-output counts come from the most recent results file of
the run, if it has them.

## Reproducing

```bash
ollama pull llama3.1:8b          # check the digest against the one in the results JSON
ollama serve
python -m app.nlp.training.train_toxicity_llm_baselines --models llama3.1:8b
```

The comparison is valid for this configuration only: one model, one prompt, two examples
per class, greedy decoding.
