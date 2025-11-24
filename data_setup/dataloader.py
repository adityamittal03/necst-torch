import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms


class RandomBitsDataset(Dataset):
    """
    random bits
    """

    def __init__(self, npy_path, transform=None):
        self.x = np.load(npy_path)
        self.transform = transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx]).float()
        if self.transform:
            x = self.transform(x)
        return x


class BinarizedMNISTDataset(Dataset):
    """
    mnist
    """

    def __init__(self, root, train=True, download=True, return_labels=False, transform=None):
        binarize = transforms.Lambda(lambda x: (x > 0.5).float())
        base_transform = transforms.Compose([transforms.ToTensor(), binarize])
        self.dataset = datasets.MNIST(
            root=str(root),
            train=train,
            download=download,
            transform=transform or base_transform,
        )
        self.return_labels = return_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.return_labels:
            return img, label
        return img


class BinarizedOmniglotDataset(Dataset):
    """
    omniglot
    """

    def __init__(self, root, background=True, download=True, return_labels=False, transform=None):
        binarize = transforms.Lambda(lambda x: (x > 0.5).float())
        base_transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(), 
            binarize
        ])
        self.dataset = datasets.Omniglot(
            root=str(root),
            background=background,
            download=download,
            transform=transform or base_transform,
        )
        self.return_labels = return_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.return_labels:
            return img, label
        return img


def load_binarized_mnist(root, val_split=10_000, download=True, return_labels=False, seed=1234):
    """
    load mnist
    """
    train_full = BinarizedMNISTDataset(
        root=root, train=True, download=download, return_labels=return_labels
    )
    train_len = len(train_full) - val_split
    if train_len <= 0:
        raise ValueError(f"Validation split {val_split} too large for dataset size {len(train_full)}")
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(train_full, [train_len, val_split], generator=generator)
    test_ds = BinarizedMNISTDataset(
        root=root, train=False, download=download, return_labels=return_labels
    )
    return train_ds, val_ds, test_ds


def load_binarized_omniglot(root, val_split=2_000, download=True, return_labels=False, seed=1234):
    """
    load omniglot
    """
    train_full = BinarizedOmniglotDataset(
        root=root, background=True, download=download, return_labels=return_labels
    )
    train_len = len(train_full) - val_split
    if train_len <= 0:
        raise ValueError(f"Validation split {val_split} too large for dataset size {len(train_full)}")
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(train_full, [train_len, val_split], generator=generator)
    test_ds = BinarizedOmniglotDataset(
        root=root, background=False, download=download, return_labels=return_labels
    )
    return train_ds, val_ds, test_ds
