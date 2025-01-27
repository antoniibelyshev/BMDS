from torch.utils.data import TensorDataset
from torch import Tensor, triu_indices


class VBEDataset(TensorDataset):
    def __init__(self, dmat_sqr: Tensor) -> None:
        idx1, idx2 = triu_indices(*dmat_sqr.shape, 1)
        super().__init__(dmat_sqr[idx1], dmat_sqr[idx2], dmat_sqr[idx1, idx2])
