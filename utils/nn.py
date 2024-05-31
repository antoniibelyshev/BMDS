from torch import nn
from typing import List


def default_create_nn_layers(
        input_dim: int,
        n_layers: int = 2,
        intermediate_dim: int = 1000,
        intermediate_activation: str = "PReLU",
        use_batch_norm: bool = True,
        output_dim: int = 100,
) -> List[nn.Module]:
    layers = []
    dims = [input_dim, [intermediate_dim] * n_layers]

    for prev_dim, next_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(prev_dim, next_dim))
        layers.append(getattr(nn, intermediate_activation)())
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(next_dim))

    layers.append(nn.Linear(dims[-1], output_dim))

    return layers
