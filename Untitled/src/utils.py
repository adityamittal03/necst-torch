"""Shared utilities for NECST Torch scripts."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
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


def infer_image_shape(input_dim: int):
    side = int(math.sqrt(input_dim))
    if side * side == input_dim:
        return (side, side)
    return (1, input_dim)


def _ensure_destination(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_reconstruction_plot(
    model,
    loader,
    device,
    destination: Path,
    n_samples: int = 8,
    shape=None,
    title: str | None = None,
):
    if loader is None:
        return
    dataset = loader.dataset
    if len(dataset) == 0:
        return
    shape = shape or infer_image_shape(dataset[0].numel())
    n_samples = min(n_samples, len(dataset))
    indices = torch.randperm(len(dataset))[:n_samples]
    batch = torch.stack([dataset[i] for i in indices]).to(device)
    model.eval()
    with torch.no_grad():
        _, _, _, _, recon_logits = model.forward_test(batch)
        recon_probs = torch.sigmoid(recon_logits)
    originals = batch.detach().cpu().view(n_samples, *shape)
    reconstructions = recon_probs.detach().cpu().view(n_samples, *shape)

    destination = _ensure_destination(destination)
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 1.4, 3), squeeze=False)
    for idx in range(n_samples):
        axes[0, idx].imshow(originals[idx], cmap="gray_r")
        axes[0, idx].set_title("input")
        axes[0, idx].axis("off")
        axes[1, idx].imshow(reconstructions[idx], cmap="gray_r", vmin=0, vmax=1)
        axes[1, idx].set_title("recon")
        axes[1, idx].axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(destination)
    plt.close(fig)
    print(f"Saved reconstruction plot to {destination}")


def save_markov_chain_plot(
    model,
    device,
    destination: Path,
    *,
    start_samples: int = 10,
    steps: int = 2000,
    interval: int = 500,
    shape=None,
):
    model.eval()
    dim = model.input_dim
    shape = shape or infer_image_shape(dim)
    x_t = torch.bernoulli(torch.full((start_samples, dim), 0.5, device=device))
    snapshots = [x_t.cpu()]
    with torch.no_grad():
        for step in range(steps):
            _, _, _, _, recon_logits = model.forward_test(x_t)
            recon_probs = torch.sigmoid(recon_logits).clamp(1e-6, 1 - 1e-6)
            x_t = torch.bernoulli(recon_probs)
            if (step + 1) % interval == 0:
                snapshots.append(x_t.cpu())

    destination = _ensure_destination(destination)
    n_steps = len(snapshots)
    n_samples = snapshots[0].shape[0]
    fig, axes = plt.subplots(n_steps, n_samples, figsize=(n_samples * 1.2, n_steps * 1.2))
    for step_idx, snapshot in enumerate(snapshots):
        for sample_idx in range(n_samples):
            ax = axes[step_idx, sample_idx] if n_steps > 1 else axes[sample_idx]
            img = snapshot[sample_idx].view(*shape)
            ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
            ax.axis("off")
            if sample_idx == 0:
                label_step = step_idx * interval if step_idx else 0
                ax.set_ylabel(f"Step {label_step}", rotation=0, labelpad=18, va="center")
    plt.tight_layout()
    plt.savefig(destination)
    plt.close(fig)
    print(f"Saved Markov chain plot to {destination}")
