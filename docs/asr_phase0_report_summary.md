# ASR Phase 0: Model Selection Findings

All models fail zero-shot on Tunisian (WER 66-93% on Maghrebi varieties).

Whisper Large v3 selected: Rank #3 Arabic leaderboard (36.9% avg WER), native Arabic output forcing via language token, largest fine-tuning community.

w2v-BERT 2.0 selected as speed alternative: RTF 0.117 (7-30x faster than Whisper), directly evaluated on TARIC-SLU Tunisian benchmark.

Ara-BEST-RQ flagged as most promising emerging model (March 2026, same team as LinTO).

Design decision: transliteration over removal for Latin-script code-switched tokens.