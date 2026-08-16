# YouTube Safe Kids — toxicity filter

Research artifact for the paper **"Toxicity Detection in Brazilian Portuguese Children's
Video Transcripts: An Annotated Corpus, a Three-Level Classifier and an Interpretable
Safety Score"**.

This repository contains the annotated corpus, the training and evaluation code, the stored
result files, and the scripts that regenerate **every figure, table and number reported in
the paper**.

> **Note on the corpus.** The published corpus (`data/corpus_toxicidade_publico.csv`)
> carries the transcribed sentences, source-video metadata and the three independent
> annotations, enabling full end-to-end reproduction of all models and experiments.

---

## 1. What the paper claims and where to check it

| Claim in the paper | Where it comes from |
|---|---|
| 1,506 sentences, 237 videos, Fleiss' $\kappa$ = 0.7693 | `analisar_corpus.py` → `results/corpus/corpus_stats.json`; also checkable directly from the published corpus file (Section 6) |
| 74.8% of toxic sentences carry no profanity; lexicon rule = 0.52 macro F1 | same script, section `decidibilidade_lexical` |
| BERTimbau 0.8946 macro F1 (cross-validation) | `train_toxicity_gridsearch.py --mode optimized` |
| BERTimbau 0.9101 macro F1 (held-out test) | `evaluate_toxicity.py` |
| TF-IDF baselines, same folds (best: SVM at 0.8320) | `train_toxicity_baselines.py` |
| Llama 3.1 8B few-shot = 0.7785 | `train_toxicity_llm_baselines.py` |
| Masked zero-shot (0.5528) and off-the-shelf classifiers (0.5430) | `train_toxicity_sem_finetuning.py` |
| McNemar and paired bootstrap | `comparar_modelos.py` |
| Operating points under both flagging rules (Tables 6 and 7) | `analisar_pontos_operacao.py` |
| Case study over 40 videos, AUC 0.888 | `analisar_estratos.py` |
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
for the winning configuration alone, one training run with 5 epochs, batch 8, learning rate
2e-5, 10% warmup, layer-wise decay 0.95, focal loss without label smoothing and **no**
oversampling. That configuration is the winner of the stability stage, not the leader of
stage 2 — the two differ, and Table 2 of the paper reports both.

Everything above is regenerable, which is why `.gitignore` excludes `models/cache/`,
`models/trained/`, `*.bin` and `*.safetensors`. Do not commit weights.

---

## 3. Reproducing the paper
 
> **Read this first.** The pipeline reads `data/corpus_toxicidade_publico.csv`, the file
> distributed here, which carries all 1,506 transcribed sentences and their annotations.
> The steps below run end-to-end from a clean clone, with no data to obtain beyond the
> model weights of Section 2. This was verified by hiding the authors' working copy and
> re-running the pipeline: the operating-point tables come out identical.

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

Two datasets are distributed here.

`data/corpus_toxicidade_publico.csv` — 1,506 annotated sentences drawn from 237 public
YouTube videos, semicolon-separated, UTF-8. **It carries the transcribed sentence itself**,
which is what makes the resource reusable rather than merely auditable: the classifier can
be retrained, the annotation inspected item by item and the Mild/Severe boundary recomputed
from the individual labels. A corpus whose sentences were withheld would support none of
that, since the annotation is attached to one specific transcription and a transcription is
not recoverable from a video identifier.

`data/reordenacao_buscas.csv` — the 40 videos of the search case study, with the query, the
position the platform returned them in, the metadata and the three sampled transcript
excerpts.

Two other files the scripts mention are deliberately absent and are not needed here.
`corpus_toxicidade.csv` is the authors' working copy of the same corpus; the pipeline falls
back to the published file when it is missing, which is the normal case in a clone.
`corpus.csv` and `corpus_analise_sentimentos.csv` belong to the Sentiment Analysis filter —
a separate study that shares this codebase, since the service is modular — and nothing in
Section 3 touches them.

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
Agreement: 77.7% unanimous, 85.1% pairwise, Fleiss' $\kappa$ = 0.7693. One sentence (0.1%)
received three distinct labels.

### What the published columns let you verify

`n_palavras` and `tem_palavrao` are computed with exactly the definitions used in the paper,
so the corpus claims can be checked from the published file without running the pipeline:

| Claim in the paper | Reproduced from |
|---|---|
| 1,506 sentences, 237 videos | row count, distinct `video_id` |
| mean length 9.0 words, median 8 | `n_palavras` |
| class balance 46.5/26.9/26.6 (700/405/401) | `tox` |
| 77.7% unanimity, 85.1% pairwise, Fleiss' $\kappa$ = 0.7693 | `p1`, `p2`, `a1` |
| 17.1% of sentences contain profanity (257) | `tem_palavrao` |
| 74.8% of toxic sentences carry none (603) | `tem_palavrao` × `tox` |
| 7.7% of non-toxic sentences do (54) | `tem_palavrao` × `tox` |
| lexicon rule reaches 0.52 macro F1 | `tem_palavrao` × `tox` |

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

The sentences were transcribed from the audio of publicly available YouTube videos. Each row
identifies its source video and carries the link, so the provenance of every annotated
sentence is traceable. No video, audio or subtitle file is redistributed: what is published
are short excerpts of speech, reproduced as quotation for research purposes.

The same holds for the case study of Section 7 of the paper. `data/reordenacao_buscas.csv`
and the stored result files `results/estratos_*.json` and `results/reordenacao_*.json` keep
every score, class, ranking and statistic **and** the 120 sampled excerpts, so
`regerar_tabelas_estudo_caso.py` rebuilds the tables and figures and
`analisar_estratos.py` reproduces the qualitative failure analysis, both from the stored
files alone.

This publishing decision has a licensing consequence, set out in Section 8: the transcribed
text is not the authors' to license, so the MIT grant does not cover it.

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

Three components, three terms — see [`LICENSE`](LICENSE) for the binding text.

| Component | Terms |
|---|---|
| Code, scripts and stored result files | MIT |
| Annotation: the three independent labels, the final label, `n_palavras`, `tem_palavrao` | CC BY 4.0 |
| Transcribed sentences, video titles and descriptions | third-party material, quoted for research; **no licence granted** |

The split is not a formality. The annotation is the authors' own work and can be licensed;
the transcribed speech belongs to the video owners, is reproduced only to the extent needed
to verify the results, and could not be placed under MIT even if that were convenient.

---

## 9. Anonymity

The journal this work targets, JBCS, operates **single-blind** review: reviewers see the
authors' identities, and the manuscript itself carries names and affiliations. This
repository therefore **does not need to be anonymous**, and making it anonymous would be
inconsistent with the submitted PDF.

If a double-blind venue is targeted later, what would need removing is limited: this file's
citation and license sections, `CITATION.cff`, and the institutional references in
`app/nlp/README.md`. No author name appears anywhere in the source code.
