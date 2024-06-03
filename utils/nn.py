from torch import nn
from typing import List


def default_create_mlp_layers(
        input_dim: int,
        intermediate_dims: List[int],
        output_dim: int,
        intermediate_activation: str,
        use_batch_norm: bool,
        last_layer_activation: bool = False,
        last_layer_batch_norm: bool = False,
) -> List[nn.Module]:
    layers = []

    for prev_dim, next_dim in zip([input_dim] + intermediate_dims, intermediate_dims):
        layers.append(nn.Linear(prev_dim, next_dim))
        layers.append(getattr(nn, intermediate_activation)())
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(next_dim))

    layers.append(nn.Linear(intermediate_dims[-1], output_dim))

    return layers
