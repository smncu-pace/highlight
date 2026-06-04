from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ImageNet-100 with parallel Hugging Face snapshot_download.")
    parser.add_argument("--repo-id", default="clane9/imagenet-100")
    parser.add_argument("--local-dir", default="data/imagenet-100-hf")
    parser.add_argument("--cache-dir", default="data/hf_snapshot_cache")
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        cache_dir=str(cache_dir),
        max_workers=args.max_workers,
        allow_patterns=["*.parquet", "README.md"],
    )
    print(f"snapshot={path}")


if __name__ == "__main__":
    main()
