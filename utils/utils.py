from typing import Union


import torch
import numpy as np


DTYPE = torch.float32


def check_tensor(t: Union[torch.Tensor, np.ndarray], dtype: torch.dtype = DTYPE) -> torch.Tensor:
    if isinstance(t, torch.Tensor):
        return t
    else:
        return torch.tensor(t, dtype=dtype)
