from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ImageNet-100 into a local Hugging Face cache.")
    parser.add_argument("--dataset", default="clane9/imagenet-100")
    parser.add_argument("--cache-dir", default="data/hf_cache")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, cache_dir=str(cache_dir))
    for split, ds in dataset.items():
        print(f"{split}: {len(ds)} examples")
    print(f"Downloaded dataset cache: {cache_dir.resolve()}")


if __name__ == "__main__":
    main()
