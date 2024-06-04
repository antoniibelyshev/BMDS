from torch.utils.data import Dataset
import torch
import numpy as np
from .preprocessing import check_tensor
from .data import NamedDataset
from typing import Union
from itertools import combinations


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DefaultBMDSDataset(NamedDataset):
    def __init__(self, dist: Union[torch.Tensor, np.ndarray], device: torch.device = DEVICE):
        dist_tensor = check_tensor(dist, device=device)
        idx = torch.tensor([*combinations(range(len(dist)), 2)])

        super().__init__(['dist', 'inp'], [dist_tensor / dist_tensor.max(), DefaultInpDataset(dist_tensor, idx)])


class DefaultInpDataset(Dataset):
    def __init__(self, dist_tensor: torch.Tensor, idx: torch.Tensor):
        super().__init__()

        self.dist_tensor = dist_tensor
        self.idx = idx

    def __len__(self):
        return self.idx.shape[0]

    def __getitem__(self, idx):
        return self.dist_tensor[self.idx[idx]]
