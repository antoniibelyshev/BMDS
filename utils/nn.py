from torch import nn


def create_mlp_layers(
        input_dim: int,
        intermediate_dims: list[int],
        output_dim: int,
        *,
        activation: str = 'ReLU',
        use_batch_norm: bool = True,
        last_layer_activation: bool = False,
        last_layer_batch_norm: bool = False,
) -> list[nn.Module]:
    layers = []

    for prev_dim, next_dim in zip([input_dim] + intermediate_dims, intermediate_dims):
        layers.append(nn.Linear(prev_dim, next_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(next_dim))
        layers.append(getattr(nn, activation)())

    layers.append(nn.Linear(intermediate_dims[-1], output_dim))
    if last_layer_batch_norm:
        layers.append(nn.BatchNorm1d(output_dim))
    if last_layer_activation:
        layers.append(getattr(nn, activation)())

    return layers
