"""Push the fine-tuned SFT model folder for TounsiLM-8b to Hugging Face Hub.

This SFT stage is trained on top of the prior CPT checkpoint:
`alabenayed/improved-aya-expanse-8b-cpt-tunisian`.

Usage:
    export HF_TOKEN=hf_...
    python llm/cpt/training/push_sft_model.py \
        --repo_id your-username/TounsiLM-8b

By default, this uploads the full fine-tuning output directory for the SFT stage:
- adapter weights
- tokenizer files
- chat template
- README model card
- training metrics/logs

You can override the default local directory with --local_dir.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, login, upload_folder


DEFAULT_LOCAL_DIR = str(
    Path(__file__).resolve().parents[3] / "outputs" / "checkpoints" / "aya-expanse-8b-tunisian-sft"
)
DEFAULT_REPO_NAME = "TounsiLM-8b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push the TounsiLM-8b SFT model folder to Hugging Face Hub")
    parser.add_argument(
        "--local_dir",
        default=DEFAULT_LOCAL_DIR,
        help="Local output directory to upload (default: the fine-tuned SFT output folder)",
    )
    parser.add_argument(
        "--repo_id",
        default=None,
        help="Target HF repo id (username/repo). If omitted, the script uses your HF username + --repo_name.",
    )
    parser.add_argument(
        "--repo_name",
        default=DEFAULT_REPO_NAME,
        help="Repository name used when --repo_id is not provided.",
    )
    parser.add_argument("--private", action="store_true", help="Create a private repo")
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="Hugging Face token (or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN env var)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.token:
        raise SystemExit("Provide HF token via --token or HF_TOKEN / HUGGINGFACE_HUB_TOKEN env var")

    login(token=args.token)
    api = HfApi()

    if args.repo_id:
        repo_id = args.repo_id
    else:
        username = api.whoami(token=args.token)["name"]
        repo_id = f"{username}/{args.repo_name}"

    api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)

    local_dir = args.local_dir
    if not Path(local_dir).exists():
        raise SystemExit(f"Local directory not found: {local_dir}")

    print(f"Uploading {local_dir} to {repo_id} ...")
    upload_folder(
        folder_path=local_dir,
        path_in_repo="",
        repo_id=repo_id,
        repo_type="model",
        token=args.token,
    )
    print(f"Upload complete: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
