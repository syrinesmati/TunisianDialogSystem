"""
Helper script to upload processed Tunisian dialect datasets to HuggingFace Hub.

Usage:
    python upload_to_huggingface.py --parquet_file processed/tunisian_arabic_clean.parquet --repo_name tunisian-dialect-corpus-cleaned --username your_username

Or programmatically:
    from upload_to_huggingface import upload_dataset
    upload_dataset(
        parquet_file="processed/tunisian_arabic_clean.parquet",
        repo_name="tunisian-dialect-corpus-cleaned",
        username="your_username"
    )
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

try:
    from huggingface_hub import HfApi, login, get_token
except ImportError:
    raise ImportError(
        "Please install huggingface-hub: pip install huggingface-hub"
    )


def upload_dataset(
    parquet_file: str,
    repo_name: str,
    username: str,
    description: str = "Cleaned Tunisian dialect corpus",
    make_private: bool = False,
    token: str = None,
    upload_metadata: bool = True,
) -> bool:
    """
    Upload a processed dataset to HuggingFace Hub.
    
    Args:
        parquet_file: Path to cleaned dataset (parquet format)
        repo_name: Repository name on HF (e.g., "tunisian-dialect-corpus-cleaned")
        username: Your HuggingFace username
        description: Dataset description
        make_private: Whether to keep dataset private
        token: HF API token (optional, uses stored token if not provided)
        upload_metadata: Whether to generate and upload metadata.json
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    parquet_path = Path(parquet_file)
    
    # Validate file exists
    if not parquet_path.exists():
        print(f"❌ File not found: {parquet_path}")
        return False
    
    print("\n" + "="*70)
    print("UPLOADING DATASET TO HUGGINGFACE HUB")
    print("="*70)

    api = HfApi()
    
    # Authenticate
    print("\n🔐 Authenticating with HuggingFace...")
    try:
        if token:
            login(token=token, add_to_git_credential_store=False)
        else:
            # Use stored token or prompt for login
            stored_token = get_token()
            if not stored_token:
                print("No stored token found. Running login...")
                login()
            else:
                print("✅ Using stored HuggingFace token")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("   Run: huggingface-cli login")
        return False

    # Preflight check: ensure uploads go to the intended HF account
    print("\n🔎 Verifying authenticated HuggingFace account...")
    try:
        whoami_info = api.whoami()
        current_user = whoami_info.get("name") or whoami_info.get("user")
        print(f"   Logged in as: {current_user}")
        print(f"   Target user:  {username}")

        if not current_user:
            print("❌ Could not determine authenticated user from HuggingFace.")
            return False

        if str(current_user).lower() != str(username).lower():
            print("❌ Username mismatch.")
            print("   The authenticated account does not match --username.")
            print("   Fix by running `huggingface-cli login` with the correct account,")
            print("   or pass the matching --username.")
            return False

        print("✅ Account verification passed")
    except Exception as e:
        print(f"❌ Could not verify authenticated user: {e}")
        return False
    
    # Get file info
    file_size_gb = parquet_path.stat().st_size / (1024**3)
    print(f"\n📄 File: {parquet_path.name}")
    print(f"   Size: {file_size_gb:.2f} GB")
    
    # Load metadata from parquet
    print(f"\n📊 Reading dataset metadata...")
    try:
        df = pd.read_parquet(parquet_path)
        num_samples = len(df)
        columns = list(df.columns)
        print(f"   Samples: {num_samples:,}")
        print(f"   Columns: {columns}")
    except Exception as e:
        print(f"⚠️  Could not read parquet: {e}")
        num_samples = None
    
    # Create/prepare repository
    print(f"\n🏗️  Creating repository...")
    repo_id = f"{username}/{repo_name}"
    
    try:
        # Create repo if it doesn't exist
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=make_private,
            exist_ok=True
        )
        print(f"✅ Repository ready: {repo_id}")
    except Exception as e:
        print(f"❌ Failed to create repository: {e}")
        return False
    
    # Upload main dataset file
    print(f"\n📤 Uploading dataset file...")
    try:
        api.upload_file(
            path_or_fileobj=str(parquet_path),
            path_in_repo=parquet_path.name,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add cleaned Tunisian corpus ({num_samples:,} samples)"
        )
        print(f"✅ Uploaded: {parquet_path.name}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False
    
    # Generate and upload metadata
    if upload_metadata and num_samples:
        print(f"\n📝 Uploading metadata...")
        metadata = {
            "dataset_name": repo_name,
            "description": description,
            "total_samples": num_samples,
            "file_size_gb": round(file_size_gb, 2),
            "uploaded_at": datetime.now().isoformat(),
            "columns": columns,
            "language": "ar-TN (Tunisian Arabic)",
            "script_type": "Arabic",
        }
        
        try:
            # Write metadata to temp file
            metadata_file = Path("metadata.json")
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Upload metadata
            api.upload_file(
                path_or_fileobj=str(metadata_file),
                path_in_repo="metadata.json",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Add dataset metadata"
            )
            print(f"✅ Uploaded: metadata.json")
            
            # Clean up temp file
            metadata_file.unlink()
        except Exception as e:
            print(f"⚠️  Could not upload metadata: {e}")
    
    # Success message
    print("\n" + "="*70)
    print("✅ UPLOAD SUCCESSFUL!")
    print("="*70)
    print(f"\n📍 Dataset URL:")
    print(f"   https://huggingface.co/datasets/{repo_id}")
    
    if make_private:
        print(f"\n🔒 Dataset is PRIVATE - only you can access it")
        print(f"   Share the link only with authorized users")
    else:
        print(f"\n🌍 Dataset is PUBLIC - anyone can find and use it")
        print(f"   Great for research community!")
    
    print(f"\n📥 To load this dataset:")
    print(f"   import pandas as pd")
    print(f'   df = pd.read_parquet("hf://datasets/{repo_id}/{parquet_path.name}")')
    
    print("\n" + "="*70 + "\n")
    
    return True


def main():
    """Command-line interface for uploading datasets."""
    parser = argparse.ArgumentParser(
        description="Upload Tunisian dialect dataset to HuggingFace Hub"
    )
    
    parser.add_argument(
        "--parquet_file",
        required=True,
        help="Path to cleaned dataset parquet file"
    )
    
    parser.add_argument(
        "--repo_name",
        required=True,
        help="Repository name on HF (e.g., 'tunisian-dialect-corpus-cleaned')"
    )
    
    parser.add_argument(
        "--username",
        required=True,
        help="Your HuggingFace username (must match authenticated account)"
    )
    
    parser.add_argument(
        "--description",
        default="Cleaned Tunisian dialect corpus",
        help="Dataset description"
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Keep dataset private (default: public)"
    )
    
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace API token (optional, uses stored token if not provided)"
    )
    
    args = parser.parse_args()
    
    success = upload_dataset(
        parquet_file=args.parquet_file,
        repo_name=args.repo_name,
        username=args.username,
        description=args.description,
        make_private=args.private,
        token=args.token,
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
