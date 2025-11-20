"""Shared utilities for NECST Torch scripts."""

from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader

from data_setup.dataloader import RandomBitsDataset, load_binarized_mnist, load_binarized_omniglot


def _load_binary_mnist(root: Path):
    # Uses torchvision MNIST with deterministic 0/1 binarization.
    return load_binarized_mnist(root, download=True, return_labels=False)


def _load_random_bits(root: Path):
    train = RandomBitsDataset(str(root / "random_bits_train.npy"))
    valid = RandomBitsDataset(str(root / "random_bits_valid.npy"))
    test = RandomBitsDataset(str(root / "random_bits_test.npy"))
    return train, valid, test


def _infer_input_dim(example) -> int:
    sample = example[0] if isinstance(example, (tuple, list)) else example
    return sample.numel()


def get_dataset_splits(dataset_name: str, datadir: str) -> Tuple[tuple, int]:
    dataset = dataset_name.lower()
    root = Path(datadir)
    if dataset in {"binarymnist", "mnist", "binary-mnist"}:
        splits = _load_binary_mnist(root / "BinaryMNIST")
    elif dataset in {"random_bits", "random", "random-bits"}:
        splits = _load_random_bits(root / "random_bits")
    elif dataset in {"omnigplot", "omniglot"}:
        splits = load_binarized_omniglot(root / "Omniglot", download=True, return_labels=False)
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'")
    input_dim = _infer_input_dim(splits[0][0])
    return splits, input_dim


def build_dataloaders(
    dataset_name: str,
    datadir: str,
    batch_size: int,
    need_train: bool = True,
    need_valid: bool = True,
    need_test: bool = True,
):
    (train_ds, valid_ds, test_ds), input_dim = get_dataset_splits(dataset_name, datadir)

    train_loader = (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True) if need_train else None
    )
    valid_loader = (
        DataLoader(valid_ds, batch_size=batch_size, shuffle=False) if need_valid else None
    )
    test_loader = (
        DataLoader(test_ds, batch_size=batch_size, shuffle=False) if need_test else None
    )
    return train_loader, valid_loader, test_loader, input_dim
