# Tunisian Arabic Spoken Dialogue System

A research project building an end-to-end spoken dialogue system for Tunisian Arabic
(Derja), a severely under-resourced dialect spoken by 12+ million people. Developed as
an INSAT end-of-year project (2025–2026).

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)

## Pipeline

```
Code-Switched Audio → Fine-tuned ASR (OmniASR-LLM-3B) → Arabic-script transcript
                                                              ↓
                        RAG-grounded response  ←  Fine-tuned LLM (TounsiLM-8b)
```

## Problem Statement

Tunisian Arabic is spoken by over 12 million people but remains severely
under-resourced. It features heavy code-switching with French, English, and Berber,
lacks a standardized orthography (Arabic script and Latin-based Arabizi coexist), and
diverges from Modern Standard Arabic in vocabulary, morphology, and syntax. This
project builds and evaluates the ASR and LLM components needed for a dialogue system
that understands and responds in authentic Tunisian Arabic.

## What's in this repo

This repo covers the **research/training side** of the project: ASR data prep,
benchmarking and fine-tuning; LLM benchmarking, continued pre-training (CPT), and
supervised fine-tuning (SFT); and the RAG knowledge-base *construction* side (schemas,
validation, embedding-text building). Two pieces described in the project's
architecture live in **separate repos** and are not included here:

- the React/Vite frontend and the ASR FastAPI serving layer (only the LLM inference
  server, `llm/serving/llm_server.py`, lives here);
- the RAG retrieval engine itself — Arabizi query rewriting/routing, hybrid
  BM25+semantic retrieval, reciprocal rank fusion, and confidence-gated context
  injection. What's here (`RAG/rag_kb/`) is the knowledge-base side: entry schemas,
  validation, and the embedding-text builder that a retrieval engine would consume.

## Architecture & Results

| Track | Component | Model | Result |
|-------|-----------|-------|--------|
| ASR | Benchmarking | 11 zero-shot candidates on Tunisian/Algerian Arabic | `facebook/omniASR-LLM-3B` selected (zero-shot WER 0.6795, RTF 0.024) |
| ASR | Fine-tuning | `facebook/omniASR-LLM-3B` on ~344h Tunisian speech | WER 0.6842 → **0.3761** (−53.1% relative), CER 0.3249 → 0.2684 |
| LLM | Benchmarking | 5 candidates on a TounsiBench-inspired pairwise benchmark | `CohereLabs/aya-expanse-8b` selected (overall score 0.61 vs. 0.54/0.43/0.38/0.23) |
| LLM | Continued pre-training | QLoRA CPT on an 85M-token Tunisian corpus (~1.18M rows) | Loss 3.633 → 2.081 (eval 2.089), no overfitting |
| LLM | Supervised fine-tuning | LoRA SFT on 31,669 instruction–response pairs | Loss 3.602 → 1.062 (−70.5%), token accuracy 42.2% → 76.2% |
| RAG | Knowledge base | 11-type sociolinguistic taxonomy across 9 source files | 1,647 validated entries (Pydantic-schema-checked) |

Final published models:

| Model | Purpose | Hugging Face |
|-------|---------|--------------|
| OmniASR-LLM-3B (fine-tuned) | ASR — Tunisian Arabic, Arabic-script output | `facebook/omniASR-LLM-3B` base, fine-tuned on `linagora/linto-dataset-audio-ar-tn` |
| improved-aya-expanse-8b-cpt-tunisian | LLM — CPT checkpoint | `alabenayed/improved-aya-expanse-8b-cpt-tunisian` |
| TounsiLM-8b | LLM — final instruction-tuned dialect model | `alabenayed/TounsiLM-8b` |

## Repo Structure

```
asr/
├── configs/                 # audio/audit/preprocessing YAML configs
├── data/lexicons/            # arabization & normalization TSV lexicons
├── src/                      # preprocessing, audit, cleaning, code-switch handling
├── pipelines/                 # run_{text,audio}_pipeline.py, run_dataset_builder.py
├── notebooks/                 # 00-04: audit → text/audio prep → dataset build → training
├── benchmarking-notebooks/     # zero-shot benchmark per candidate model (10+ models)
├── benchmarking-csv/           # benchmark result CSVs
├── training/
│   ├── omni-asr-3b-finetuning.py   # the fine-tuning run actually used
│   └── legacy/                      # Whisper/w2v-BERT fine-tuning scripts, not selected
└── evaluation/                # WER/CER/RTF evaluation

llm/
├── benchmarking/               # 5-model TounsiBench-inspired pairwise benchmark
├── cpt/
│   ├── data/scripts/            # corpus collection, cleaning, HF upload
│   └── training/                # train.py (CPT), sft_train_question_response.py (SFT),
│                                  push_adapter.py, push_sft_model.py, quick_test.py
├── sft/data/
│   ├── raw/, cleaned.csv, augmented/, final/   # SFT dataset pipeline stages
│   ├── scripts/                  # exploration, cleaning, augmentation notebooks
│   └── legacy/                    # superseded dataset snapshot + notebook variant
└── serving/
    └── llm_server.py             # FastAPI SSE server for TounsiLM-8b

RAG/
└── rag_kb/
    ├── data/            # 9 JSON knowledge-base source files
    ├── schemas/          # Pydantic entry schemas (BaseEntry, ExpressionEntry, ProverbEntry)
    ├── pipeline/          # build_embed_text.py, validate_entries.py
    ├── scripts/            # bulk_import.py
    └── db/chroma_db/       # vector store (populated at index time)
```

## Setup

```bash
git clone <this-repo-url>
cd TunisianDialogSystem
pip install -r requirements.txt
```

`omnilingual-asr` (ASR training/inference) and `vibevoice` (one legacy ASR benchmark
script) are installed from source — see the comments at the bottom of
`requirements.txt`.

## Usage

**ASR — zero-shot benchmark:** notebooks under `asr/benchmarking-notebooks/`.

**ASR — fine-tune the selected model:**
```bash
python asr/training/omni-asr-3b-finetuning.py
```

**ASR — evaluate:**
```bash
python asr/evaluation/evaluate_asr.py
```

**LLM — benchmark candidate base models:**
```bash
python llm/benchmarking/run_benchmark.py
```

**LLM — continued pre-training (CPT):**
```bash
python llm/cpt/training/train.py
```

**LLM — supervised fine-tuning (SFT):**
```bash
python llm/cpt/training/sft_train_question_response.py
```

**LLM — serve TounsiLM-8b:**
```bash
uvicorn llm.serving.llm_server:app --host 0.0.0.0 --port 8000
```
Defaults to the published `alabenayed/TounsiLM-8b` model; set `MODEL_DIR` to point at
a local checkpoint/adapter directory instead.

**RAG — validate/build the knowledge base:**
```python
from RAG.rag_kb.pipeline.validate_entries import validate_file, print_validation_report
from RAG.rag_kb.pipeline.build_embed_text import build_embed_text
```

## Datasets

- **LinTO**: ~400h Tunisian Arabic speech corpus (CC-BY 4.0, `linagora/linto-dataset-audio-ar-tn`)
- **Tunisian Dialect Corpus**: ~1.18M rows / 85M tokens raw Tunisian text for CPT (`Syrinesmati/tunisian-dialect-corpus`)
- **Tunisian Q/A dataset**: 31,669 instruction–response pairs for SFT (`Syrinesmati/tunisian-question-response-dataset`)

## Team

Smati Syrine · Ben Ayed Mohamed Ala · Sassi Yasmine
Supervisor: Mrs. Hajer Taktak — INSAT, Academic Year 2025–2026

## License

MIT — see [LICENSE](LICENSE).
