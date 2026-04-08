# System Architecture

The Tunisian Arabic Spoken Dialogue System consists of two main tracks: Automatic Speech Recognition (ASR) and Large Language Model (LLM), integrated into an end-to-end pipeline.

## ASR Track

- **Input**: Code-switched audio (Tunisian Arabic with French/English/Berber)
- **Preprocessing**: Transliteration of Latin-script tokens to Arabic phonetics
- **Model**: Fine-tuned Whisper Large v3 or w2v-BERT 2.0
- **Output**: Arabic-script transcript

## LLM Track

- **Input**: Arabic-script transcript
- **Model**: Fine-tuned aya-expanse-8b with QLoRA
- **Output**: Tunisian Arabic response

## Integration Pipeline

The pipeline combines ASR and LLM for real-time dialogue, with options for streaming inference.