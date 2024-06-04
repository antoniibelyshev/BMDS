import torch
import numpy as np
from typing import Union, Any


def check_tensor(t: Union[torch.Tensor, np.ndarray], *, dtype: torch.dtype = torch.float32, **kwargs: Any):
    return torch.tensor(t, dtype=dtype, **kwargs)
