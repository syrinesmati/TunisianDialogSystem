from setuptools import setup, find_packages

setup(
    name="tunisian-dialogue-system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchaudio",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "jiwer",
        "anthropic",
        "pandas",
        "numpy",
        "tqdm",
        "pyyaml",
        "soundfile",
        "librosa",
        "evaluate",
        "sentencepiece",
        "sacrebleu",
    ],
    author="Syrine Smati, Mohamed Ala Ben Ayed, Yasmine Sassi",
    description="End-to-end spoken dialogue system for Tunisian Arabic",
)