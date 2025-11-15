import numpy as np
import torch
from torch.utils.data import Dataset


class RandomBitsDataset(Dataset):
    """
    Loads binary vectors stored as .npy files (train/valid/test splits).
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


class BinaryMNISTDataset(Dataset):
    """
    Loads binarized MNIST stored in .amat format (one sample per line).
    """

    def __init__(self, amat_path, transform=None):
        self.x = np.loadtxt(amat_path)
        self.transform = transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx]).float()
        if self.transform:
            x = self.transform(x)
        return x
