# Tunisian Arabic Spoken Dialogue System

A research project building an end-to-end spoken dialogue system for Tunisian Arabic (Darija), a severely under-resourced dialect spoken by 12+ million people.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)

## Pipeline Diagram

```
Code-Switched Audio → Fine-tuned ASR Model → Arabic-Script Transcript → Fine-tuned LLM → Tunisian Arabic Response
```

## Problem Statement

Tunisian Arabic is spoken by over 12 million people but remains severely under-resourced. It features heavy code-switching with French, English, and Berber, and lacks standardized orthography. This project addresses the gap by building an end-to-end dialogue system that can understand and respond in authentic Tunisian Arabic.

## Architecture

| Track | Component | Status | Description |
|-------|-----------|--------|-------------|
| ASR | Model Selection | ✅ | Whisper Large v3 selected |
| ASR | Fine-tuning | 🔄 | Fine-tuning on LinTO corpus |
| LLM | Benchmarking | ✅ | aya-expanse-8b selected |
| LLM | Fine-tuning | 📋 | QLoRA fine-tuning planned |
| Integration | Pipeline | 📋 | End-to-end integration planned |

## Project Phases

- **ASR Phase 0: Model selection** ✅ — Whisper Large v3 selected (Rank #3 Arabic leaderboard, 36.9% avg WER)
- **ASR Phase 1: Fine-tuning** 🔄 — Fine-tuning Whisper Large v3 + w2v-BERT 2.0 on LinTO 400h corpus
- **LLM Phase 1: Benchmarking** ✅ — aya-expanse-8b selected
- **LLM Phase 2: QLoRA fine-tuning** 📋 — of aya-expanse-8b
- **Phase 3: System integration** 📋

## Key Models

| Model | Purpose | Rationale |
|-------|---------|-----------|
| Whisper Large v3 | ASR primary | Rank #3 Arabic leaderboard, native Arabic output forcing |
| w2v-BERT 2.0 | ASR alternative | 7-30x faster inference (RTF 0.117) |
| aya-expanse-8b | LLM base | Best correctness+relevance, 23+ language multilingual coverage |
| TunBERT | Dialect verifier | Encoder-only, used for data filtering and output scoring |

## Datasets

- **LinTO**: ~400h Tunisian Arabic speech corpus (CC-BY 4.0, linagora/linto-dataset-audio-ar-tn-0.1-augmented)
- **TounsiBench**: 744 gold instruction pairs (EMNLP 2025)

## Expected Outcomes

- ASR WER: 25–40% (vs 66–93% zero-shot)
- LLM dialectal adherence: 1.5–1.8/2.0 (vs <1.0 current open-source)

## Installation

```bash
git clone https://github.com/yourusername/tunisian-dialogue-system.git
cd tunisian-dialogue-system
pip install -r requirements.txt
```

## Usage

### Download Data

```bash
python data/scripts/download_linto.py
```

### Fine-tune ASR

```bash
python asr/training/finetune_whisper.py --config configs/whisper_finetune_config.yaml
```

### Fine-tune LLM

```bash
python llm/training/finetune_aya_qlora.py --config configs/qlora_config.yaml
```

### Run Dialogue Pipeline

```bash
python pipeline/dialogue_pipeline.py --audio input.wav
```

## References

- TounsiBench: EMNLP 2025
- Open Universal Arabic ASR Leaderboard: Interspeech 2025
- LinTO: arXiv 2025

## Project Status

- [x] ASR Phase 0: Model selection
- [x] LLM Phase 1: Benchmarking
- [ ] ASR Phase 1: Fine-tuning
- [ ] LLM Phase 2: QLoRA fine-tuning
- [ ] Phase 3: System integration

**Team:** Syrine Smati, Mohamed Ala Ben Ayed, Yasmine Sassi — Academic Year 2025/2026