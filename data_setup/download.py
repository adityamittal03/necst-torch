"""
Utility script to fetch/generate datasets.

Allowed datasets: {BINARYMNIST, random_bits, omnigplot}
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on the path when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_setup.dataloader import load_binarized_mnist


def download_binary_mnist(dest: Path, val_split: int = 10_000):
    """
    Download torchvision MNIST data and materialize the binarized train/val/test splits.
    """
    dest.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, test_ds = load_binarized_mnist(dest, val_split=val_split, download=True)
    print(f"Binary MNIST downloaded under: {dest}")
    print(f"Splits -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")


def generate_random_bits(dest: Path, n_bits: int):
    """
    Generate synthetic random bit vectors with fixed split sizes.
    """
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)

    train = rng.integers(0, 2, size=(5000, n_bits), dtype=np.int64)
    valid = rng.integers(0, 2, size=(1000, n_bits), dtype=np.int64)
    test = rng.integers(0, 2, size=(1000, n_bits), dtype=np.int64)

    np.save(dest / "random_bits_train.npy", train)
    np.save(dest / "random_bits_valid.npy", valid)
    np.save(dest / "random_bits_test.npy", test)
    print(f"Random bits saved to: {dest} (n_bits={n_bits})")


def main():
    parser = argparse.ArgumentParser(description="Download/generate datasets for NECST.")
    parser.add_argument("dataset", type=str, help="Dataset name (case-insensitive).")
    parser.add_argument("--datadir", type=str, default="data", help="Root directory for datasets")
    parser.add_argument(
        "--n-bits",
        type=int,
        default=100,
        help="Bit-width for random_bits dataset (ignored for other datasets).",
    )
    args = parser.parse_args()

    dataset = args.dataset.lower()
    root = Path(args.datadir)

    if dataset in {"binarymnist", "mnist"}:
        download_binary_mnist(root / "BinaryMNIST")
    elif dataset == "random_bits":
        generate_random_bits(root / "random_bits", args.n_bits)
    elif dataset in {"omnigplot", "omniglot"}:
        from data_setup.dataloader import load_binarized_omniglot

        dest = root / "Omniglot"
        dest.mkdir(parents=True, exist_ok=True)
        train_ds, val_ds, test_ds = load_binarized_omniglot(dest, download=True, return_labels=False)
        print(f"Omniglot downloaded under: {dest}")
        print(f"Splits -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    else:
        raise ValueError(f"Unknown dataset '{args.dataset}'")


if __name__ == "__main__":
    main()
