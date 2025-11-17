"""Shared utilities for NECST Torch scripts."""

from pathlib import Path

from torch.utils.data import DataLoader

from data_setup.dataloader import BinaryMNISTDataset, RandomBitsDataset


def _load_binary_mnist(root: Path):
    train = BinaryMNISTDataset(str(root / "binarized_mnist_train.amat"))
    valid = BinaryMNISTDataset(str(root / "binarized_mnist_valid.amat"))
    test = BinaryMNISTDataset(str(root / "binarized_mnist_test.amat"))
    return train, valid, test


def _load_random_bits(root: Path):
    train = RandomBitsDataset(str(root / "random_bits_train.npy"))
    valid = RandomBitsDataset(str(root / "random_bits_valid.npy"))
    test = RandomBitsDataset(str(root / "random_bits_test.npy"))
    return train, valid, test


def get_dataset_splits(dataset_name: str, datadir: str):
    dataset = dataset_name.lower()
    root = Path(datadir)
    if dataset in {"binarymnist", "mnist"}:
        splits = _load_binary_mnist(root / "BinaryMNIST")
    elif dataset in {"random_bits", "random", "random-bits"}:
        splits = _load_random_bits(root / "random_bits")
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'")
    input_dim = splits[0][0].numel()
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

