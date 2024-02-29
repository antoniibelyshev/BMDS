from typing import Union, Tuple
import pytorch_lightning as pl
import torch
from torch.utils import data
from itertools import combinations, product


class Dataset(data.Dataset):
    tensors: Tuple[torch.Tensor, ...]
    stds: Tuple[torch.Tensor, ...]

    def __init__(
            self,
            *tensors: torch.Tensor,
            stds: Tuple[torch.Tensor, ...] = (),
    ):
        super().__init__()

        self.tensors = tensors
        self.stds = stds

    def __len__(self) -> int:
        return len(self.tensors[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return tuple(self.get_kth(k, idx) for k in range(len(self.tensors)))

    def get_kth(self, k: int, idx: int) -> torch.Tensor:
        if k < len(self.stds):
            return self.tensors[k][idx] + torch.randn_like(self.stds[k][idx]) * self.stds[k][idx]
        else:
            return self.tensors[k][idx]


class DefaultDataModule(pl.LightningDataModule):
    dataset: Dataset
    batch_size: int

    def __init__(
            self,
            *tensors: torch.Tensor,
            batch_size: Union[int, float] = 1e-2,
            stds: Tuple[torch.Tensor, ...] = (),
    ):
        super().__init__()

        self.dataset = Dataset(*tensors, stds=stds)

        if isinstance(batch_size, float):
            batch_size = int(len(self.dataset) * batch_size)
        self.batch_size = batch_size

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


class DistMatrixDataModule(DefaultDataModule):
    def __init__(
            self,
            dist_mat: torch.Tensor,
            *,
            batch_size: Union[int, float] = 1e-2,
            symmetric: bool = True,
    ):
        n, m = dist_mat.shape
        idx = torch.tensor(list(combinations(range(n), 2) if symmetric else product(range(n), range(m))))
        super().__init__(idx, (dist_mat[tuple(idx.T)] / dist_mat.max()) ** 2, batch_size=batch_size)
