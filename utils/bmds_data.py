from torch.utils.data import Dataset
import torch
import numpy as np
from .preprocessing import check_tensor
from .data import NamedDataset
from typing import Union
from itertools import combinations


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DefaultBMDSDataset(NamedDataset):
    def __init__(self, dist: Union[torch.Tensor, np.ndarray], *, device: torch.device = DEVICE):
        self.dist_tensor = check_tensor(dist, device=device)
        idx = self.get_idx()

        super().__init__(['dist', 'inp'], [self.dist_tensor[idx.T[0], idx.T[1]] / self.dist_tensor.max(), DefaultInpDataset(self.dist_tensor, idx)])

    def get_idx(self):
        return torch.tensor([*combinations(range(len(self.dist_tensor)), 2)])


class NeighborsDataset(Dataset):
    def __init__(self, neighbors_idx: torch.Tensor, dist: torch.Tensor):
        super().__init__()

        self.neighbors_idx = neighbors_idx
        self.dist = check_tensor(dist)

        n, self.m = neighbors_idx.shape
        self.l = n * self.m

        self.items_idx = torch.stack((torch.arange(n).repeat(self.m, 1).T.reshape(-1), neighbors_idx.reshape(-1)), 1)

    def __getitem__(self, idx: int):
        row = idx // self.m
        col = idx % self.m
        return {'neighbors_idx': self.neighbors_idx[[row, col]], 'neighbors_dist': self.dist[[row, col]], 'dist': self.dist[row, col]}

    def __len__(self):
        return self.l


def neighbors_collate_fn(batch, n_obj, dtype=torch.float32):
    n, m = len(batch), batch[0]['neighbors_idx'].shape[1]
    # inp = torch.sparse_csr_tensor(
    #     crow_indices=torch.arange(2 * n) * m,
    #     col_indices=torch.stack([b['neighbors_idx'] for b in batch]).reshape(-1),
    #     values=torch.stack([b['neighbors_idx'] for b in batch]).reshape(-1),
    #     dtype=dtype,
    #     size=(2 * n, n_obj),
    # )
    inp = {
        'neighbors_idx': torch.stack([b['neighbors_idx'] for b in batch]).reshape(-1, m),
        'dist': torch.stack([b['neighbors_dist'] for b in batch]).reshape(-1, m),
    }
    dist = torch.tensor([b['dist'] for b in batch])
    return {'inp': inp, 'dist': dist}


class DefaultNeighborsBMDSDataset(DefaultBMDSDataset):
    def __init__(
        self,
        dist: Union[torch.Tensor, np.ndarray],
        n_neighbors: Union[int, float] = 100,
        *,
        device: torch.device = DEVICE,
        sparse_input: bool = False
    ):
        if isinstance(n_neighbors, float):
            n_neighbors = int(n_neighbors * len(dist))
        self.n_neighbors = n_neighbors

        self.sparse_input = sparse_input

        super().__init__(dist, device=device)

    def get_idx(self):
        objects_idx = torch.arange(len(self.dist_tensor)).repeat(self.n_neighbors, 1).reshape(-1)
        neighbors_idx = self.dist_tensor.argsort(dim=0)[-self.n_neighbors:].reshape(-1)
        idx = torch.stack((objects_idx, neighbors_idx), dim=1)

        if self.sparse_input:
            dist_tensor, self.dist_tensor = self.dist_tensor, torch.zeros_like(self.dist_tensor)
            self.dist_tensor[idx.T[0], idx.T[1]] = dist_tensor[idx.T[0], idx.T[1]]

        return idx


class DefaultInpDataset(Dataset):
    def __init__(self, dist_tensor: torch.Tensor, idx: torch.Tensor, sparse: bool = False):
        super().__init__()

        self.dist_tensor = dist_tensor
        self.idx = idx
        self.sparse = sparse

    def __len__(self):
        return self.idx.shape[0]

    def __getitem__(self, idx):
        if self.sparse:
            return self.dist_tensor[self.idx[idx]].to_sparse()
        else:
            return self.dist_tensor[self.idx[idx]]
