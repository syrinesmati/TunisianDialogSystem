# Dataset Strategy and Sources

## LinTO Corpus

- **Source**: linagora/linto-dataset-audio-ar-tn-0.1-augmented
- **Size**: ~400 hours
- **License**: CC-BY 4.0
- **Usage**: ASR fine-tuning
- **Characteristics**: Tunisian Arabic speech with code-switching

## TounsiBench

- **Source**: EMNLP 2025
- **Size**: 744 gold instruction pairs
- **Usage**: LLM fine-tuning and evaluation
- **Characteristics**: High-quality Tunisian Arabic dialogue pairs

## Synthetic Data Generation

- **Method**: Use Claude API to generate Tunisian pairs
- **Filtering**: TunBERT dialect scorer
- **Merging**: Combine with gold data for training