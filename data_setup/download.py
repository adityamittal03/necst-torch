"""
Allowed datasets: {binarymnist, random_bits, omnigplot}
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Ensure repo root is on the path when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description="Download datasets for Necst-Torch.")
    parser.add_argument("dataset", type=str, help="Dataset name (case-insensitive).")
    parser.add_argument("--datadir", type=str, default="data", help="Root directory for datasets")
    parser.add_argument(
        "--n-bits",
        type=int,
        default=100,
        help="Bit-width for random_bits dataset.",
    )
    args = parser.parse_args()

    dataset = args.dataset.lower()
    root = Path(args.datadir)

    # binaryMNIST
    if dataset in {"binarymnist", "mnist"}:
        from data_setup.dataloader import load_binarized_mnist
        dest = root / "BinaryMNIST"
        dest.mkdir(parents=True, exist_ok=True)
        train_ds, val_ds, test_ds = load_binarized_mnist(dest, val_split=10_000, download=True)
        print(f"Binary MNIST downloaded under: {dest}")
        print(f"Splits -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    # random bits
    elif dataset == "random_bits":
        dest = root / "random_bits"
        dest.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(1234)

        train = rng.integers(0, 2, size=(5000, args.n_bits), dtype=np.int64)
        valid = rng.integers(0, 2, size=(1000, args.n_bits), dtype=np.int64)
        test = rng.integers(0, 2, size=(1000, args.n_bits), dtype=np.int64)

        np.save(dest / "random_bits_train.npy", train)
        np.save(dest / "random_bits_valid.npy", valid)
        np.save(dest / "random_bits_test.npy", test)
        print(f"Random bits saved to: {dest} (n_bits={args.n_bits})")

    # omniglot
    elif dataset == "omniglot":
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
