from torch import nn
from typing import List


def create_mlp_layers(
        input_dim: int,
        intermediate_dims: List[int],
        output_dim: int,
        activation: str = 'ReLU',
        use_batch_norm: bool = True,
        last_layer_activation: bool = False,
        last_layer_batch_norm: bool = False,
) -> List[nn.Module]:
    layers = []

    for prev_dim, next_dim in zip([input_dim] + intermediate_dims, intermediate_dims):
        layers.append(nn.Linear(prev_dim, next_dim))
        layers.append(getattr(nn, activation)())
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(next_dim))

    layers.append(nn.Linear(intermediate_dims[-1], output_dim))
    if last_layer_activation:
        layers.append(getattr(nn, activation)())
    if last_layer_batch_norm:
        layers.append(nn.BatchNorm1d(output_dim))

    return layers
