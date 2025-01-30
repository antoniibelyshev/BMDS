from torch.utils.data import TensorDataset
from torch import Tensor, triu_indices


class VBEDataset(TensorDataset):
    def __init__(self, dmat_sqr: Tensor) -> None:
        idx1, idx2 = triu_indices(*dmat_sqr.shape, 1)
        super().__init__(idx1, idx2)
        self.dmat_sqr = dmat_sqr

    def __getitem__(self, idx: int) -> tuple[Tensor]:
        idx1, idx2 = super().__getitem__(idx)
        return self.dmat_sqr[idx1], self.dmat_sqr[idx2], self.dmat_sqr[idx1, idx2]
