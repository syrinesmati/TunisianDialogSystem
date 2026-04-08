# LLM Phase 1: Benchmarking Findings

All 5 models fail to produce authentic Tunisian in zero-shot settings.

Rankings: aya-expanse-8b (1st, best correctness+relevance), Labess-7b (2nd, inconsistent), SILMA-9B (3rd, understands Tunisian but outputs MSA), LLaMA-3-8B (4th), Llama-3.2-1B (5th, insufficient capacity).

Root bottleneck: data scarcity — only 744 gold instruction pairs exist publicly (TounsiBench).

Mitigation: synthetic data generation + controlled translation + LinTO augmentation + TunBERT filtering.

QLoRA selected for memory efficiency (fits ≤24GB VRAM).

Expected outcome after fine-tuning: dialectal adherence 1.5-1.8/2.0 (approaching GPT-4o-mini reference of 1.68).