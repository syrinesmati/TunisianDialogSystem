#!/usr/bin/env python3
"""Standalone runner for the Microsoft VibeVoice-ASR Tunisian Arabic ASR benchmark.

This script mirrors the Whisper large-v3 benchmark runner logic so it can be
launched from tmux or another long-running server session without losing
progress when SSH drops.

Key differences from the Whisper runner
----------------------------------------
- Uses the `vibevoice` package (cloned from https://github.com/microsoft/VibeVoice
  and installed with `pip install -e .`) instead of HuggingFace `pipeline()`.
- No language/task flag needed — the model auto-detects language.
- The processor expects `audio=<file_path>`, so each in-memory example is
  written to a temp WAV, transcribed, then the file is deleted.
- Output is Rich Transcription markup; `post_process_transcription()` turns it
  into plain text for WER/CER scoring.
- batch_size defaults to 1 (7B model).
- dtype: bfloat16 on CUDA, float32 on MPS/CPU.
- attn: flash_attention_2 on CUDA, sdpa elsewhere.

Prerequisites
-------------
  git clone https://github.com/microsoft/VibeVoice
  cd VibeVoice && pip install -e .
  # ffmpeg must be available on $PATH

Outputs
-------
- per-split CSV files in the configured benchmark output directory
- summary CSV and WER/CER chart
- full stdout/stderr transcript in a log file alongside the outputs
"""

from __future__ import annotations

# ============================================================
# STAGE 0 — absolute first thing: prove we are actually running
# ============================================================
import sys
import os
import time
import json

print("=" * 60, flush=True)
print("run_vibevoice_asr_benchmark.py  — starting", flush=True)
print(f"Python  : {sys.version}", flush=True)
print(f"PID     : {os.getpid()}", flush=True)
print("=" * 60, flush=True)

import argparse
import io
import re
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Microsoft VibeVoice-ASR benchmark as a plain Python script."
    )
    parser.add_argument(
        "--dataset-root",
        default="/home/ala/dataset",
        help="Path to the dataset root that contains config.yaml and the split folders.",
    )
    parser.add_argument(
        "--output-root",
        default="/home/ala/TunisianDialogSystem/outputs/asr_benchmark_results",
        help="Root directory for benchmark outputs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints if available.",
    )
    parser.add_argument(
        "--context-info",
        default="",
        help=(
            "Optional hotwords / context hint forwarded to VibeVoice-ASR "
            "(comma-separated names, technical terms, etc.)."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Hard cap for generated tokens per audio sample (default: 2048).",
    )
    parser.add_argument(
        "--tokens-per-second",
        type=float,
        default=20.0,
        help="Dynamic token budget multiplier based on audio duration (default: 20).",
    )
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=256,
        help="Minimum token budget per sample for dynamic cap (default: 256).",
    )
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help=(
            "'auto' picks flash_attention_2 on CUDA and sdpa on MPS/CPU."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for inference (overrides config.yaml if set; default: 1 from config).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging helper — tee stdout/stderr to a log file
# ---------------------------------------------------------------------------

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def install_console_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    print(f"Logging to: {log_path}", flush=True)


# ---------------------------------------------------------------------------
# VibeVoice inference wrapper
# ---------------------------------------------------------------------------

def build_vibevoice_infer_fn(
    model_id: str,
    context_info: str,
    target_sr: int,
    batch_size: int,
    max_new_tokens: int,
    tokens_per_second: float,
    min_new_tokens: int,
    attn_implementation: str,
):
    """Return an infer_fn(ds) compatible with benchmark_utils.

    Loads the model once. Returns a callable that transcribes a HuggingFace
    Dataset and returns a list of plain-text strings.
    """
    import numpy as np
    import soundfile as sf
    import torch

    print("[VibeVoice] Importing vibevoice package ...", flush=True)
    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
    print("[VibeVoice] vibevoice imported OK", flush=True)

    # ---- device & dtype (mirrors upstream demo) ----------------------------
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        attn_impl = "flash_attention_2" if attn_implementation == "auto" else attn_implementation
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
        attn_impl = "sdpa" if attn_implementation == "auto" else attn_implementation
    else:
        device = "cpu"
        dtype = torch.float32
        attn_impl = "sdpa" if attn_implementation == "auto" else attn_implementation

    print(
        f"[VibeVoice] device={device}  dtype={dtype}  attn={attn_impl}",
        flush=True,
    )

    # ---- load processor ----------------------------------------------------
    print(f"[VibeVoice] Loading processor from '{model_id}' ...", flush=True)
    processor = VibeVoiceASRProcessor.from_pretrained(model_id)
    print("[VibeVoice] Processor loaded", flush=True)

    # ---- load model (mirrors upstream demo device handling) ----------------
    print(f"[VibeVoice] Loading model from '{model_id}' ...", flush=True)
    if device == "mps":
        # MPS: device_map="mps" is not supported; load on CPU then move
        model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=None,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        model = model.to("mps")
    else:
        model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"[VibeVoice] ✓ Model ready  ({total_params / 1e9:.2f}B params on {device})",
        flush=True,
    )

    # ---- helpers -----------------------------------------------------------

    _markup_re = re.compile(r"<\|[^|]*?\|>")

    def _get_audio_array(example) -> np.ndarray:
        audio = example.get("audio") or example.get("speech")
        if audio is None:
            raise KeyError(
                f"No 'audio' or 'speech' key. Keys: {list(example.keys())}"
            )
        # Handle multiple possible audio representations stored in the dataset
        # - HuggingFace `Audio` dict: {'array': ..., 'sampling_rate': ...}
        # - HuggingFace/torchcodec AudioDecoder object (from cast_column)
        # - dict with 'path': path to file
        # - raw numpy array
        # - string path
        arr = None
        sr = None

        # HF datasets torchcodec decoder path
        if hasattr(audio, "get_all_samples") and callable(audio.get_all_samples):
            samples = audio.get_all_samples()
            arr = samples.data
            sr = int(samples.sample_rate)

        if isinstance(audio, dict):
            if "array" in audio:
                arr = audio["array"]
                sr = audio.get("sampling_rate")
            elif "bytes" in audio and audio["bytes"] is not None:
                # Datasets may store embedded audio as raw bytes with no path.
                arr, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            elif "path" in audio:
                if audio["path"] is None:
                    raise ValueError("Audio dict has no usable 'array', 'bytes', or 'path'.")
                arr, sr = sf.read(audio["path"], dtype="float32")  # returns (data, sr)
            else:
                # try to find any array-like value in the dict
                for v in audio.values():
                    if hasattr(v, "__array__") or isinstance(v, (list, tuple)):
                        arr = v
                        break
                if arr is None:
                    raise KeyError(f"No audio array found in dict keys: {list(audio.keys())}")
        elif isinstance(audio, (str, Path)):
            arr, sr = sf.read(str(audio))
        else:
            # assume it's already an array-like object
            arr = audio

        # Unwrap lazy/nested audio wrappers (e.g. AudioDecoder in dict['array']).
        for _ in range(4):
            changed = False

            if hasattr(arr, "get_all_samples") and callable(arr.get_all_samples):
                samples = arr.get_all_samples()
                arr = samples.data
                if sr is None and hasattr(samples, "sample_rate"):
                    sr = int(samples.sample_rate)
                changed = True

            elif isinstance(arr, dict):
                if "array" in arr:
                    arr = arr["array"]
                    if sr is None:
                        sr = arr.get("sampling_rate") if isinstance(arr, dict) else sr
                    changed = True
                elif "bytes" in arr and arr["bytes"] is not None:
                    arr, sr = sf.read(io.BytesIO(arr["bytes"]), dtype="float32")
                    changed = True
                elif "path" in arr and arr["path"] is not None:
                    arr, sr = sf.read(arr["path"], dtype="float32")
                    changed = True

            elif isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
                changed = True

            elif not isinstance(arr, (list, tuple)) and not hasattr(arr, "__array__"):
                # try common attributes/methods found on lazy audio objects
                if hasattr(arr, "array"):
                    arr = arr.array
                    changed = True
                elif hasattr(arr, "to_array"):
                    arr = arr.to_array()
                    changed = True
                elif hasattr(arr, "decode"):
                    try:
                        arr = arr.decode()
                        changed = True
                    except Exception:
                        pass
                elif hasattr(arr, "read"):
                    try:
                        arr = arr.read()
                        changed = True
                    except Exception:
                        pass

            if not changed:
                break

        arr = np.asarray(arr, dtype=np.float32)

        # If multi-channel, convert to mono by averaging channel axis.
        if arr.ndim > 1:
            try:
                # For shape (samples, channels), average axis=1.
                # For shape (channels, samples), average axis=0.
                channel_axis = 1 if arr.shape[1] <= arr.shape[0] else 0
                arr = arr.mean(axis=channel_axis)
            except Exception:
                arr = arr.reshape(-1)

        # Resample if needed (prefer librosa if available)
        if sr is not None and sr != target_sr:
            try:
                import librosa

                arr = librosa.resample(arr.astype(np.float32), orig_sr=sr, target_sr=target_sr)
            except Exception:
                # fallback: simple numpy-based resample (not high quality)
                import math

                old_len = arr.shape[0]
                new_len = int(math.ceil(old_len * float(target_sr) / sr))
                arr = np.interp(
                    np.linspace(0, old_len - 1, new_len),
                    np.arange(old_len),
                    arr,
                ).astype(np.float32)

        return arr

    def _write_temp_wav(audio_array: np.ndarray, sr: int) -> str:
        """Persist array to a temp WAV file; return path."""
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        audio_int16 = (audio_array * 32768.0).clip(-32768, 32767).astype(np.int16)
        sf.write(tmp.name, audio_int16, sr, subtype="PCM_16")
        return tmp.name

    def _raw_to_plain(raw_text: str) -> str:
        """Convert VibeVoice output to plain text without emitting parser warnings."""

        def _strip_role_prefix(text: str) -> str:
            text = text.strip()
            if text.startswith("assistant"):
                text = text[len("assistant"):].lstrip(" \t:\n\r")
            return text

        def _balanced_json_slice(text: str) -> str:
            for open_ch, close_ch in (("[", "]"), ("{", "}")):
                start = text.find(open_ch)
                end = text.rfind(close_ch)
                if start != -1 and end != -1 and end > start:
                    return text[start : end + 1]
            return text

        def _extract_text(obj) -> str:
            pieces: list[str] = []

            def walk(x):
                if isinstance(x, dict):
                    for key in ("Content", "content", "text", "transcript", "sentence", "utterance"):
                        val = x.get(key)
                        if isinstance(val, str) and val.strip():
                            pieces.append(val.strip())
                            return
                    for val in x.values():
                        walk(val)
                elif isinstance(x, list):
                    for item in x:
                        walk(item)
                elif isinstance(x, str):
                    if x.strip():
                        pieces.append(x.strip())

            walk(obj)
            return " ".join(pieces)

        cleaned = _strip_role_prefix(raw_text)

        # Try to parse assistant-wrapped JSON or JSON-like payloads.
        for candidate in (cleaned, _balanced_json_slice(cleaned)):
            try:
                parsed = json.loads(candidate)
                plain = _extract_text(parsed)
                if plain.strip():
                    return " ".join(plain.split())
            except Exception:
                pass

        # Regex fallback for quoted Content fields.
        content_matches = re.findall(r'"Content"\s*:\s*"(.*?)"', cleaned, flags=re.DOTALL)
        if content_matches:
            joined = " ".join(content_matches)
            return " ".join(joined.replace('\\"', '"').split())

        # Fallback: strip all <|…|> tokens
        return " ".join(_markup_re.sub(" ", cleaned).split())

    # ---- main callable -----------------------------------------------------

    def infer_fn(ds):
        # Ensure the dataset audio column is a HuggingFace Audio column with arrays
        try:
            from datasets import Audio as HFAudio
            ds = ds.cast_column("audio", HFAudio(sampling_rate=target_sr))
        except Exception:
            # if casting fails, continue — _get_audio_array handles multiple formats
            pass

        results: list[str] = []
        latencies: list[float] = []
        n = len(ds)
        infer_start = time.time()

        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_t0 = time.time()

            for idx in range(batch_start, batch_end):
                example = ds[idx]
                audio_array = _get_audio_array(example)
                tmp_path = _write_temp_wav(audio_array, target_sr)

                try:
                    inputs = processor(
                        audio=tmp_path,
                        sampling_rate=target_sr,
                        return_tensors="pt",
                        add_generation_prompt=True,
                        context_info=context_info if context_info else None,
                    )
                    inputs = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in inputs.items()
                    }

                    duration_s = max(float(len(audio_array)) / float(target_sr), 0.0)
                    dynamic_cap = int(duration_s * float(tokens_per_second)) + int(min_new_tokens)
                    effective_max_new_tokens = max(int(min_new_tokens), min(int(max_new_tokens), dynamic_cap))

                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=effective_max_new_tokens,
                            do_sample=False,
                            temperature=1.0,
                            top_p=1.0,
                            pad_token_id=processor.pad_id,
                            eos_token_id=processor.tokenizer.eos_token_id,
                        )

                    # Keep only newly generated tokens (strip input prefix)
                    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
                    
                    # Try raw decode first (without processor post-processing)
                    try:
                        # Use tokenizer directly to avoid processor post-processing bugs
                        raw_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    except Exception as decode_exc:
                        try:
                            # Fallback: use processor decode
                            raw_text = processor.decode(generated_ids, skip_special_tokens=True)
                        except Exception as decode_exc2:
                            print(f"  [VibeVoice] ⚠ Sample {idx}: Both decode methods failed: {decode_exc2}", flush=True)
                            raw_text = ""
                    
                    # Debug: log raw output if it's very short or empty
                    if len(raw_text.strip()) < 20:
                        print(f"  [VibeVoice] ⚠ Sample {idx}: Raw output very short or empty: '{raw_text}'", flush=True)
                    
                    plain_text = _raw_to_plain(raw_text)

                except Exception as exc:
                    print(f"  [VibeVoice] ⚠ Error on example {idx}: {exc}", flush=True)
                    plain_text = ""

                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

                results.append(plain_text)

            batch_elapsed = time.time() - batch_t0
            per_sample = batch_elapsed / max(batch_end - batch_start, 1)
            latencies.extend([per_sample] * (batch_end - batch_start))

            print(
                f"  [VibeVoice] transcribed {batch_end}/{n} examples",
                flush=True,
            )

        return results, latencies, time.time() - infer_start

    return infer_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("main() entered", flush=True)

    args = parse_args()
    print(f"Args: {args}", flush=True)

    dataset_root = Path(args.dataset_root)
    output_root  = Path(args.output_root)

    # ---- STAGE 1: validate paths ------------------------------------------
    print(f"[Stage 1] Checking dataset root: {dataset_root}", flush=True)
    if not dataset_root.exists():
        print(f"ERROR: dataset root does not exist: {dataset_root}", flush=True)
        return 1
    print(f"[Stage 1] dataset root OK", flush=True)

    config_yaml = dataset_root / "config.yaml"
    print(f"[Stage 1] Checking config.yaml: {config_yaml}", flush=True)
    if not config_yaml.exists():
        print(f"ERROR: config.yaml not found at {config_yaml}", flush=True)
        return 1
    print(f"[Stage 1] config.yaml OK", flush=True)

    # ---- STAGE 2: sys.path + env ------------------------------------------
    print(f"[Stage 2] Adding dataset root to sys.path", flush=True)
    if str(dataset_root) not in sys.path:
        sys.path.insert(0, str(dataset_root))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # ---- STAGE 3: import benchmark_utils ----------------------------------
    print("[Stage 3] Importing benchmark_utils ...", flush=True)
    try:
        from benchmark_utils import (
            audio_inspector,
            display_bulk_predictions,
            display_preview,
            display_summary,
            display_worst,
            get_device,
            load_config,
            plot_wer_cer,
            print_gpu_info,
            run_labelled_splits,
            run_unlabelled_splits,
            setup_output_dir,
            split_benchmark,
        )
    except ImportError as exc:
        print(f"ERROR importing benchmark_utils: {exc}", flush=True)
        print(f"sys.path = {sys.path}", flush=True)
        return 1
    print("[Stage 3] benchmark_utils imported OK", flush=True)

    # ---- STAGE 4: import datasets -----------------------------------------
    print("[Stage 4] Importing datasets ...", flush=True)
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        print(f"ERROR importing datasets: {exc}", flush=True)
        return 1
    print("[Stage 4] datasets imported OK", flush=True)

    # ---- STAGE 5: load config ---------------------------------------------
    print("[Stage 5] Loading config.yaml ...", flush=True)
    cfg = load_config(str(config_yaml))
    cfg["paths"]["dataset"]     = str(dataset_root)
    cfg["paths"]["output_root"] = str(output_root)
    print(f"[Stage 5] Config loaded. Keys: {list(cfg.keys())}", flush=True)

    target_sr    = cfg["evaluation"]["target_sr"]
    top_n_worst  = cfg["evaluation"]["top_n_worst"]
    preview_rows = cfg["evaluation"]["preview_rows"]

    model_key  = "vibevoice_asr"
    mcfg       = cfg.get("models", {}).get(model_key, {})
    model_id   = mcfg.get("model_id",   "microsoft/VibeVoice-ASR")
    batch_size = mcfg.get("batch_size", 1)
    
    # CLI override for batch_size
    if args.batch_size is not None:
        batch_size = args.batch_size
        print(f"[Stage 5] CLI override: batch_size = {batch_size}", flush=True)

    # If the config.yaml doesn't include a `vibevoice_asr` model entry,
    # create a minimal one so `setup_output_dir` and other helpers work.
    if "models" not in cfg:
        cfg["models"] = {}
    if model_key not in cfg["models"]:
        print(f"[Stage 5] Notice: '{model_key}' not found in config.yaml — creating a default entry", flush=True)
        cfg["models"][model_key] = {
            "model_id": model_id,
            "output_subdir": "vibevoice_results",
            "batch_size": batch_size,
            "chunk_length_s": 30,
            "stride_length_s": 5,
            "torch_dtype": "bfloat16",
        }
        mcfg = cfg["models"][model_key]

    # ---- STAGE 6: setup output dir + logging ------------------------------
    print("[Stage 6] Setting up output directory ...", flush=True)
    benchmark_output_dir = Path(setup_output_dir(cfg, model_key))
    log_path = benchmark_output_dir / f"{model_key}_run.log"
    install_console_logging(log_path)

    # From here everything is also written to the log file.
    print("=" * 60)
    print("Dataset path        :", cfg["paths"]["dataset"])
    print("Output root         :", cfg["paths"]["output_root"])
    print("Model key           :", model_key)
    print("Model id            :", model_id)
    print("Batch size          :", batch_size)
    print("Max new tokens      :", args.max_new_tokens)
    print("Tokens per second   :", args.tokens_per_second)
    print("Min new tokens      :", args.min_new_tokens)
    print("Attention impl      :", args.attn_implementation)
    print("Context info        :", args.context_info or "(none)")
    print("Resume              :", args.resume)
    print("=" * 60)

    get_device()
    print_gpu_info()

    output_dir = setup_output_dir(cfg, model_key)

    # ---- STAGE 7: load model ----------------------------------------------
    print("[Stage 7] Building VibeVoice inference function ...", flush=True)
    infer_fn = build_vibevoice_infer_fn(
        model_id=model_id,
        context_info=args.context_info,
        target_sr=target_sr,
        batch_size=batch_size,
        max_new_tokens=args.max_new_tokens,
        tokens_per_second=args.tokens_per_second,
        min_new_tokens=args.min_new_tokens,
        attn_implementation=args.attn_implementation,
    )
    print("[Stage 7] Inference function ready", flush=True)

    # ---- STAGE 8: load dataset splits ------------------------------------
    print("[Stage 8] Loading dataset splits ...", flush=True)
    split_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    benchmark  = DatasetDict()
    for split_dir in split_dirs:
        try:
            benchmark[split_dir.name] = load_from_disk(str(split_dir))
            print(f"  Loaded split: {split_dir.name}", flush=True)
        except Exception as exc:
            print(f"  Skipping {split_dir.name}: {exc}", flush=True)

    print(f"[Stage 8] Splits loaded: {list(benchmark.keys())}", flush=True)
    labelled_splits, unlabelled_splits = split_benchmark(benchmark)
    print(f"  labelled   : {labelled_splits}", flush=True)
    print(f"  unlabelled : {unlabelled_splits}", flush=True)

    # ---- STAGE 9: run labelled splits ------------------------------------
    notebook_splits = ["labeled_linagora_raw"]
    print(f"[Stage 9] Running only labeled_linagora_raw: {notebook_splits}", flush=True)

    all_result_dfs, summary_rows = run_labelled_splits(
        benchmark,
        notebook_splits,
        infer_fn,
        output_dir,
        model_key,
        preview_rows,
        top_n_worst,
        resume=args.resume,
    )
    print("[Stage 9] Labelled splits done", flush=True)

    display_preview(all_result_dfs, preview_rows)
    display_worst(all_result_dfs, top_n_worst)

    # ---- STAGE 10: run unlabelled splits ---------------------------------
    print("[Stage 10] Running unlabelled splits ...", flush=True)
    unlabelled_result_dfs = run_unlabelled_splits(
        benchmark,
        unlabelled_splits,
        infer_fn,
        output_dir,
        model_key,
        resume=args.resume,
    )
    print("[Stage 10] Unlabelled splits done", flush=True)

    if unlabelled_splits:
        audio_inspector(
            benchmark,
            unlabelled_result_dfs,
            unlabelled_splits,
            target_sr=target_sr,
        )
        display_bulk_predictions(unlabelled_result_dfs)

    # ---- STAGE 11: summary & chart ---------------------------------------
    print("[Stage 11] Generating summary ...", flush=True)
    summary_df = display_summary(
        summary_rows,
        output_dir,
        model_key,
        model_id,
    )
    if summary_df is not None:
        plot_wer_cer(summary_df, output_dir, model_key, model_id)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())