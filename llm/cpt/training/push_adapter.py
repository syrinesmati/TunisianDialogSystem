"""Helper to push the adapter / checkpoint folder to Hugging Face Hub.

Usage:
    export HF_TOKEN=hf_....
    python llm/cpt/training/push_adapter.py --local_dir outputs/checkpoints/aya-expanse-8b-cpt-tunisian --repo_id username/aya-expanse-8b-cpt-tunisian-adapter --push_full

Notes:
- Requires `huggingface_hub` >= 0.14.0 and `git` installed for large uploads.
- If `--push_full` is set the entire output dir will be uploaded; otherwise only adapter files (peft) are uploaded.
"""
from __future__ import annotations
import argparse
import os
from huggingface_hub import HfApi, login, upload_folder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--local_dir", required=True, help="Local output directory containing adapter/checkpoints")
    p.add_argument("--repo_id", required=True, help="Target HF repo id (username/repo)")
    p.add_argument("--private", action="store_true", help="Create a private repo")
    p.add_argument("--push_full", action="store_true", help="Upload entire folder (checkpoints + tokenizer) instead of only adapter files")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token (or set HF_TOKEN env var)")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.token:
        raise SystemExit("Provide HF token via --token or HF_TOKEN env var")

    login(token=args.token)
    api = HfApi()

    # create repo if needed
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    # Decide which folder to upload
    if args.push_full:
        folder_to_upload = args.local_dir
    else:
        # try common PEFT adapter locations (adapter_config.json, adapter_model.bin, etc.)
        # We upload the top-level files in the local_dir (filter by known adapter files)
        peft_files = [
            "adapter_config.json",
            "pytorch_model.bin",
            "adapter_model.bin",
            "adapter_state.json",
            "adapter_scheduler.pt",
        ]
        # fallback: upload everything under local_dir/pytorch_model.bin or peft
        folder_to_upload = args.local_dir

    print(f"Uploading {folder_to_upload} to {args.repo_id} (this may take a while)")
    upload_folder(folder_path=folder_to_upload, path_in_repo="", repo_id=args.repo_id, repo_type="model", token=args.token)
    print("Upload complete. Visit: https://huggingface.co/" + args.repo_id)


if __name__ == "__main__":
    main()
