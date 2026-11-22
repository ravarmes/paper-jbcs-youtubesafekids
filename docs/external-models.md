# External models: identifiers, revisions, training data and licences

Every model this repository uses but did not train. Facts about each model come from its
public model card and the Hugging Face API, checked on **2026-11-19**. URLs are pinned to the
revision used, so they keep pointing at the same files if the repositories change.

## Summary

| Role in the paper | Identifier | Revision used (commit) | Training data (per card) | Licence (per card) |
|---|---|---|---|---|
| Base model for fine-tuning, linear probe and masked zero-shot | [`neuralmind/bert-base-portuguese-cased`](https://huggingface.co/neuralmind/bert-base-portuguese-cased/tree/94d69c95f98f7d5b2a8700c420230ae10def0baa) | `94d69c95f98f7d5b2a8700c420230ae10def0baa` | brWaC | MIT |
| Off-the-shelf offensiveness classifier | [`ruanchaves/bert-base-portuguese-cased-hatebr`](https://huggingface.co/ruanchaves/bert-base-portuguese-cased-hatebr/tree/ed31720b2f3c0b6cefe81cf267751b255b377a7d) | `ed31720b2f3c0b6cefe81cf267751b255b377a7d` | HateBR (`ruanchaves/hatebr`) | none stated for the model; the card restricts the HateBR **dataset** to academic and research use |
| Off-the-shelf target classifier | [`dougtrajano/toxicity-target-classification`](https://huggingface.co/dougtrajano/toxicity-target-classification/tree/885d2698f9b68f0fc90850fe7fc4dc7cc970f2f5) | `885d2698f9b68f0fc90850fe7fc4dc7cc970f2f5` | OLID-BR (`dougtrajano/olid-br`) | Apache-2.0 |
| Generative baseline | `llama3.1:8b` (Ollama) | see `docs/llm-baseline.md` | — | Llama 3.1 Community License |

For all three Hugging Face models the revision used is also the current head of `main`
(API field `sha` on 2026-11-19), so "latest" and "used" coincide today. They may not in the
future; the pinned commit is what reproduces the paper.

## How the revision used was established

The two off-the-shelf classifiers are not loaded from the Hub at run time. They are read
from local copies in `data/models/`, downloaded with `hf download --local-dir`. That command
leaves `.cache/huggingface/download/<file>.metadata` next to each file, whose first line is
the commit and second line the file's etag (for LFS files, the sha256 of the content):

| Copy | Commit in every `.metadata` | `pytorch_model.bin` sha256, from metadata | sha256 recomputed locally |
|---|---|---|---|
| `data/models/ruanchaves_bert_base_hatebr` | `ed31720b…377a7d` | `8447edf46bf471477005e336896a5a13f67880b41c3a2059966343897c2a1e7c` | identical |
| `data/models/dougtrajano_toxicity_target_classification` | `885d2698…c970f2f5` | `2494ada141fc9b43ee71dd958e772766aedc9a020ce7e6f7b3a5de7cbcc44948` | identical |

The base model is loaded through the Hugging Face cache in `app/nlp/models/cache/`, whose
`refs/main` points at `94d69c95f98f7d5b2a8700c420230ae10def0baa`.

**Why the scripts do not pass `revision=`.** `from_pretrained` receives a local directory
for the two classifiers, where `revision` has no effect, so there is nothing to pin in the
call. Instead, `train_toxicity_sem_finetuning.py` reads the commit from the `.metadata`
files and writes it to its results JSON (`modelos_externos`: identifier, URL, local folder,
revision used, expected revision, whether they match, weights sha256, label set), and it
logs a warning if a local copy is from a different commit than the one above. It also
records the base model's cached revision (`modelo_base`). The pinning therefore lives in
the download step:

```bash
hf download ruanchaves/bert-base-portuguese-cased-hatebr \
    --revision ed31720b2f3c0b6cefe81cf267751b255b377a7d \
    --local-dir data/models/ruanchaves_bert_base_hatebr
hf download dougtrajano/toxicity-target-classification \
    --revision 885d2698f9b68f0fc90850fe7fc4dc7cc970f2f5 \
    --local-dir data/models/dougtrajano_toxicity_target_classification
```

Both checkpoints contain exactly the standard `bert.*` and `classifier.*` weights (202
tensors each), so `BertForSequenceClassification` loads them completely, even though the
target classifier's `config.json` names a custom architecture
(`ToxicityTypeForSequenceClassification`).

## `ruanchaves/bert-base-portuguese-cased-hatebr`

- **What it is:** BERTimbau base (`neuralmind/bert-base-portuguese-cased`) fine-tuned for
  offensive-language detection on HateBR. Created 2023-03-27; last modified 2023-04-13.
- **Labels** (card and `config.json`): `0` = the text is not offensive (`false`), `1` = the
  text is offensive (`true`).
- **Training data:** HateBR (`ruanchaves/hatebr` on the Hub). The paper cites the dataset as
  `vargas2022hatebr` (expert-annotated Instagram comments in Brazilian Portuguese).
- **Licence:** the model card has no licence field. Its "Licensing Information" section
  states that the HateBR dataset "is provided strictly for academic and research purposes"
  and that any commercial or non-academic use is prohibited without prior written consent
  from SINCH.
- **Citation the card asks for** (the authors' repository, pending a paper):

```bibtex
@software{Chaves_Rodrigues_eplm_2023,
author = {Chaves Rodrigues, Ruan and Tanti, Marc and Agerri, Rodrigo},
doi = {10.5281/zenodo.7781848},
month = {3},
title = {{Evaluation of Portuguese Language Models}},
url = {https://github.com/ruanchaves/eplm},
version = {1.0.0},
year = {2023}
}
```

## `dougtrajano/toxicity-target-classification`

- **What it is:** BERTimbau base fine-tuned to classify "if a given text is targeted or
  not", on the OLID-BR dataset. Created 2023-02-11; last modified 2023-02-15. Input: Brazilian
  Portuguese text; output: binary.
- **Labels** (`config.json`): `0` = `UNTARGETED`, `1` = `TARGETED INSULT`. The model card
  gives no definitions; the OLID-BR dataset card defines them: `TIN` Targeted Insult —
  "targeted insult or threat towards an individual, a group or other"; `UNT` Untargeted —
  "non-targeted profanity and swearing". In OLID-BR the target of a targeted insult may be an
  individual (`IND`), a group (`GRP`) or **other** (`OTH`: "an organization, an event, an
  issue, etc.").
- **Training data:** OLID-BR (`dougtrajano/olid-br`), 7,943 comments in v1.0 "from different
  sources, including social media (YouTube and Twitter) and related datasets" (dataset card;
  dataset licence CC BY 4.0).
- **Performance reported on the card** (OLID-BR test set): accuracy 0.6864, precision
  0.6882, recall 0.6864, F1 0.6872; per class, `UNTARGETED` F1 0.4961 (support 443) and
  `TARGETED INSULT` F1 0.7723 (support 995).
- **Licence:** Apache-2.0.
- **Citation:** the model card asks for none. The OLID-BR dataset card asks for the dataset
  paper, which is already in `latex/referencias.bib` as `trajano2024olidbr`
  (Trajano, Bordini and Vieira, *Language Resources and Evaluation* 58(4):1263–1289, 2024).

## Detoxify, `multilingual` model

- **What it is:** XLM-RoBERTa base fine-tuned by Unitary on the Jigsaw multilingual toxicity
  data; returns independent probabilities for toxicity, severe toxicity, obscene, identity
  attack, insult, threat and sexual explicit. Used only as a reference that does not train
  on this corpus (`app/nlp/evaluation/scripts/avaliar_detoxify.py`).
- **Package:** `detoxify==0.5.3` (with `sentencepiece==0.2.0`), both in `requirements.txt`.
  On Windows, `sentencepiece` 0.2.2 crashes the process (access violation in
  `msvcp140.dll`) when imported after scikit-learn, whose wheel bundles an older C++
  runtime; 0.2.0 does not. Keep the pinned version.
- **Checkpoint used:** `multilingual_debiased-0b549669.ckpt`, from
  `https://github.com/unitaryai/detoxify/releases/download/v0.4-alpha/`, 1,112,295,725
  bytes, sha256
  `0b5496691403977207f24778695389494a412a76c2cb351e99f024f2a4e3f3ff` (recomputed locally on
  2026-11-20). The library loads it with `torch.hub.load_state_dict_from_url`, and takes the
  configuration and tokenizer of `xlm-roberta-base` from the Hugging Face Hub.
- **Citation:** `hanu2020detoxify` in `latex/referencias.bib` (DOI 10.5281/zenodo.7925667).

## `neuralmind/bert-base-portuguese-cased` (BERTimbau base)

Revision `94d69c95f98f7d5b2a8700c420230ae10def0baa` (last modified 2022-06-14); card
metadata: licence MIT, training data brWaC. Used as the starting point of the fine-tuned
classifier, as the frozen encoder of the linear probe, and as the masked language model
of the zero-shot reference.

## Notes for the manuscript (reviewer comment M6c)

These are observations from the cards, for the authors to decide on; they are not claims
of the paper.

1. **Neither off-the-shelf classifier was trained on ToLD-BR.** The offensiveness model
   comes from HateBR and the target model from OLID-BR. A sentence saying that ToLD-BR is the
   reference for the transfer measured with these models would not match what was run.
2. **The target notions differ.** In this corpus, Mild covers criticism aimed at objects,
   situations or abstract entities; in OLID-BR, an insult aimed at "an organization, an
   event, an issue" is *targeted* (`OTH`). Mapping `TARGETED INSULT` to Severe therefore
   sends part of what this corpus calls Mild to Severe by construction.
3. **The target classifier is weak on its own data for the class that maps to Mild:**
   `UNTARGETED` F1 0.4961 on the OLID-BR test set, per its card.
