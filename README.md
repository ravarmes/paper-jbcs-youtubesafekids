# YouTube Safe Kids — toxicity filter

Research artifact for the paper **"Toxicity Detection in Brazilian Portuguese Children's
Video Transcripts: An Annotated Corpus, a Three-Level Classifier and an Interpretable
Safety Score"**.

This repository contains the annotated corpus, the training and evaluation code, the stored
result files, and the scripts that regenerate **every figure, table and number reported in
the paper**.

> **Note on the corpus.** The transcribed sentences are **not redistributed**. The
> published corpus (`data/corpus_toxicidade_publico.csv`) carries the annotation, the
> source-video identifiers and the derived measurements, but not the text. Section 6
> explains what this does and does not allow.

---

## 1. What the paper claims and where to check it

| Claim in the paper | Where it comes from |
|---|---|
| 1,506 sentences, 251 videos, Fleiss' $\kappa$ = 0.7795 | `analisar_corpus.py` → `results/corpus/corpus_stats.json`; also checkable directly from the published corpus file (Section 6) |
| 77.4% of toxic sentences carry no profanity; lexicon rule = 0.50 macro F1 | same script, section `decidibilidade_lexical` |
| BERTimbau 0.9220 macro F1 (cross-validation) | `train_toxicity_gridsearch.py --mode optimized` |
| BERTimbau 0.8778 macro F1 (held-out test) | `evaluate_toxicity.py` |
| TF-IDF baselines, same folds | `train_toxicity_baselines.py` |
| Llama 3.1 8B few-shot = 0.6930 | `train_toxicity_llm_baselines.py` |
| Masked zero-shot and off-the-shelf classifiers | `train_toxicity_sem_finetuning.py` |
| McNemar and paired bootstrap | `comparar_modelos.py` |
| Case study over 40 videos, AUC 0.835 | `analisar_estratos.py` |
| Sampling simulation, 30,000 videos per cell | `simular_amostragem.py` |

Everything cited in the manuscript is consolidated in
`app/nlp/evaluation/results/artigo/resumo_numeros.json`, which is the single file to
consult when auditing a number.

---

## 2. Requirements

- Python 3.9+
- A YouTube Data API key (only for the live service and the search case study; **not**
  needed to reproduce the model results)
- A GPU is strongly recommended for fine-tuning; the search costs 159 training runs

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then fill in YOUTUBE_API_KEY
```

### Model weights to download

**No model weights ship with this repository**, and none are in version control: they are
either downloaded from third parties or produced by training. About **7 GB** in total, but
you only need what the step you are running asks for.

| Weights | Identifier | Size | Needed by |
|---|---|---:|---|
| BERTimbau Base (cased) — the backbone that is fine-tuned | `neuralmind/bert-base-portuguese-cased` | 438 MB | steps 2, 4, 5 |
| HateBR offensive-language classifier | `ruanchaves/bert-base-portuguese-cased-hatebr` | 436 MB | step 5 |
| Toxicity target classifier (directed vs. undirected) | `dougtrajano/toxicity-target-classification` | 436 MB | step 5 |
| Llama 3.1 8B, served locally | `ollama pull llama3.1:8b` | 4.9 GB | step 6 |
| BERTimbau Large — *optional*, only if you change the backbone | `neuralmind/bert-large-portuguese-cased` | 1.3 GB | — |

**BERTimbau Base** downloads itself on first use: `from_pretrained` writes it to
`app/nlp/models/cache/`, the path set by `cache_dir` in `app/nlp/config.py`. Nothing to do
in advance, but the first run needs network access.

**The two off-the-shelf classifiers of step 5 do not download themselves.** The script
reads them from local directories, so fetch them once:

```bash
pip install "huggingface_hub[cli]"
hf download ruanchaves/bert-base-portuguese-cased-hatebr \
    --local-dir data/models/ruanchaves_bert_base_hatebr
hf download dougtrajano/toxicity-target-classification \
    --local-dir data/models/dougtrajano_toxicity_target_classification
```

The directory names matter: `app/nlp/training/train_toxicity_sem_finetuning.py` looks for
exactly those two paths under `data/models/`.

**Llama 3.1 8B** runs through [Ollama](https://ollama.com), locally and offline — no cloud
API is involved. Install Ollama, then `ollama pull llama3.1:8b` and `ollama serve`. The
host can be overridden with `OLLAMA_HOST` (default `http://localhost:11434`).

**The fine-tuned classifier** is not downloadable: it is the output of step 2, written to
`app/nlp/models/trained/`. Reproducing it means running the search (159 training runs) or,
for the winning configuration alone, one training run with 8 epochs, batch 8, learning rate
5e-5, 6% warmup, layer-wise decay 0.9 and cross-entropy with 0.1 label smoothing.

Everything above is regenerable, which is why `.gitignore` excludes `models/cache/`,
`models/trained/`, `*.bin` and `*.safetensors`. Do not commit weights.

---

## 3. Reproducing the paper

> **Read this first.** The pipeline reads `data/corpus_toxicidade.csv`, the corpus
> **including** the `FRASE` column with the transcribed sentences. That file is **not
> distributed** (Section 6): what ships here is `data/corpus_toxicidade_publico.csv`, which
> has the annotation but not the text. The steps below therefore cannot be run end to end
> from the published files alone. What the published file does support is listed in
> Section 6: every corpus-level claim in the paper can be verified from it.

Run from this directory, in this order. Steps 1–7 need network access only for the model
downloads listed in Section 2.

```bash
# 1. corpus statistics, Fleiss' kappa, and the lexical-decidability measurements
python -m app.nlp.evaluation.scripts.analisar_corpus

# 2. hyperparameter search (random search + 3-stage Successive Halving) and final model
python -m app.nlp.training.train_toxicity_gridsearch --mode optimized

# 3. classical baselines over exactly the same folds
python -m app.nlp.training.train_toxicity_baselines

# 4. evaluation on the held-out test set
python -m app.nlp.evaluation.evaluate_toxicity

# 5. references that do not use these labels (masked zero-shot, off-the-shelf classifiers)
python -m app.nlp.training.train_toxicity_sem_finetuning

# 6. generative baseline (requires Ollama running)
python -m app.nlp.training.train_toxicity_llm_baselines

# 7. significance tests (McNemar, paired bootstrap)
python -m app.nlp.evaluation.scripts.comparar_modelos

# 8. worked example of the safety score
python -m app.nlp.evaluation.scripts.calcular_scores_exemplo

# 9. case study over real searches (needs the trained model)
python -m app.nlp.evaluation.scripts.analisar_estratos --idioma en

# 10. sampling simulation
python -m app.nlp.evaluation.scripts.simular_amostragem
```

### Figures and tables

The manuscript is in English, so the generators take `--idioma en`. `--duas-colunas`
emits `table*` instead of `table`, which the two-column journal template requires.

```bash
python -m app.nlp.evaluation.scripts.gerar_figuras_artigo \
    --idioma en --duas-colunas --copiar-para ../latex/figuras
python -m app.nlp.evaluation.scripts.regerar_tabelas_estudo_caso --idioma en --duas-colunas
python -m app.nlp.evaluation.scripts.gerar_figura_conceitual --idioma en
```

`regerar_tabelas_estudo_caso.py` rebuilds the case-study outputs from the stored JSON
files, so it works **without** the trained model — useful for checking the tables without
re-running step 2.

Dropping `--idioma en` regenerates everything in Portuguese.

---

## 4. Running the service

```bash
uvicorn app.main:app --reload
```

The web service re-ranks results returned by the YouTube Data API; it does not crawl the
platform, does not store personal data and does not modify the underlying content. A
`YOUTUBE_API_KEY` is required.

---

## 5. Layout

```
app/
  api/            HTTP endpoints
  core/           configuration, YouTube client
  filters/        Duration, Engagement, Sentiment, Toxicity
  nlp/
    datasets/     corpus preparation and splitting
    models/       BERTimbau wrappers (weights are not versioned)
    training/     hyperparameter search, baselines, LLM and zero-shot references
    evaluation/   held-out evaluation
      scripts/    corpus analysis, case study, simulation, figures and tables
      results/    stored result files -- the evidence behind every number
  static/, templates/   web interface
data/             the annotated corpus
docs/             experiment notes and a grid-search notebook
scripts/          standalone utilities (preprocessing, kappa, reordering test)
```

`app/nlp/README.md` documents the NLP module in more depth, in Portuguese.

---

## 6. Data

`data/corpus_toxicidade_publico.csv` — 1,506 annotated sentences drawn from 251 public
YouTube videos, semicolon-separated, UTF-8. **It is the only dataset in this repository.**

Scripts reference three other data files, all deliberately absent. `corpus_toxicidade.csv`
is this same corpus *with* the sentences: the steps of Section 3 read it, and it is not
distributed for the reasons below. `corpus.csv` and `corpus_analise_sentimentos.csv` belong
to the Sentiment Analysis filter — a separate study that shares this codebase, since the
service is modular — and nothing in Section 3 touches them.

**The transcribed sentences themselves are not redistributed.** What is published is the
annotation and everything needed to audit the corpus claims, minus the text. See
*Provenance and redistribution* below for what this means in practice.

| Column | Meaning |
|---|---|
| `id_sentenca` | stable row identifier, 1–1506 |
| `video_id` | YouTube video identifier |
| `titulo` | video title, as published |
| `parte` | which part of the video the sentence came from: `Inicio`, `Meio`, `Fim` |
| `link` | video URL |
| `n_palavras` | length of the sentence in words |
| `tem_palavrao` | 1 if the sentence matches the profanity lexicon used in the paper, else 0 |
| `tox` | **final label**, the mode of the three annotations |
| `p1`, `p2`, `a1` | the three independent annotations, pseudonymised |

Labels: `Nenhuma` (None) — appropriate; `Leve` (Mild) — indirect offensiveness, aimed at
objects, situations or abstract entities; `Severa` (Severe) — direct offensiveness, aimed
at a person or group. The distinction that separates Mild from Severe is the **target** of
the offence, not its intensity.

Three annotators labelled every sentence independently; the final label is the mode.
Agreement: 79.1% unanimous, 86.0% pairwise, Fleiss' $\kappa$ = 0.7795.

### What the published columns let you verify

Although the text is withheld, `n_palavras` and `tem_palavrao` are computed with exactly
the definitions used in the paper, so the corpus claims can be checked directly from the
published file:

| Claim in the paper | Reproduced from |
|---|---|
| 1,506 sentences, 251 videos | row count, distinct `video_id` |
| mean length 8.8 words, median 7 | `n_palavras` |
| class balance 50/25/25 | `tox` |
| 79.1% unanimity, 86.0% pairwise, Fleiss' $\kappa$ = 0.7795 | `p1`, `p2`, `a1` |
| 17.1% of sentences contain profanity | `tem_palavrao` |
| 77.4% of toxic sentences carry none | `tem_palavrao` × `tox` |
| 11.6% of non-toxic sentences do | `tem_palavrao` × `tox` |
| lexicon rule reaches 0.50 macro F1 | `tem_palavrao` × `tox` |

What the published file does **not** allow is retraining or re-evaluating the classifier:
that needs the sentences.

### Notes on the data

- Sentences are transcriptions of **speech from public videos**, obtained with an external
  transcription tool. Transcription artefacts are present in the annotated data and were
  deliberately preserved rather than edited, since editing would alter the string the
  annotators actually read.
- The corpus holds **no personal data** about children or caregivers. Video identifiers,
  titles and URLs are public metadata.
- Annotator columns are pseudonymous.
- Every sentence is distinct; there are no duplicates.

### Provenance and redistribution

The sentences were transcribed from the audio of publicly available YouTube videos. The
authors chose **not to redistribute the transcribed text**, and to publish the annotation
together with the derived measurements above instead. Each row still identifies its source
video, so the provenance of every annotated sentence is traceable.

This is a deliberate trade-off, and its cost is stated plainly here and in the paper:
because the annotation is attached to a specific transcription, and a transcription is not
recoverable from a video identifier — re-transcribing with a different tool yields
different strings — **the published file does not allow the model results to be
reproduced**, only the corpus-level claims to be verified.

`app/nlp/datasets/gerar_corpus_publico.py` is the script that derives the published file
from the full corpus; it documents exactly what was removed and what was computed.

The same rule was applied to the case study of Section 7 of the paper. The stored result
files `results/estratos_*.json` and `results/reordenacao_*.json` keep every score, class,
ranking and statistic, but their `frases` fields are **blank**: the 120 sampled excerpts
are transcriptions of third-party video audio, exactly like the corpus sentences, and are
withheld for the same reason. The excerpts quoted in the paper are quoted there because
they are needed to substantiate specific claims. One consequence is worth stating:
`regerar_tabelas_estudo_caso.py` rebuilds the case-study tables and figures from these
files unchanged, but `analisar_estratos.py` can no longer reproduce the qualitative
failure analysis from the stored files alone — that needs re-running the collection.

---

## 7. Citing

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff); GitHub renders it as a
*Cite this repository* button. The manuscript is under review at the *Journal of the
Brazilian Computer Society*: once it has a DOI, add it to `CITATION.cff` and here.

```bibtex
@misc{santos2026youtubesafekids,
  author = {Vargas Mesquita Santos, Rafael and Louzada da Silva, Gabriel and
            Vargas, Sabrina and Pedrosa, Daniela and Maroquio Bernardo, Ricardo},
  title  = {{YouTube Safe Kids --- toxicity filter: corpus and code}},
  year   = {2026},
  url    = {https://github.com/ravarmes/paper-jbcs-youtubesafekids}
}
```

---

## 8. License

**MIT** — see [`LICENSE`](LICENSE). This covers the code and the published corpus file in
`data/`, which contains the authors' own annotation and derived measurements rather than
any transcribed content from the platform.

---

## 9. Anonymity

The journal this work targets, JBCS, operates **single-blind** review: reviewers see the
authors' identities, and the manuscript itself carries names and affiliations. This
repository therefore **does not need to be anonymous**, and making it anonymous would be
inconsistent with the submitted PDF.

If a double-blind venue is targeted later, what would need removing is limited: this file's
citation and license sections, `CITATION.cff`, and the institutional references in
`app/nlp/README.md`. No author name appears anywhere in the source code.
