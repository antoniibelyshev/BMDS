from torch import Tensor
from torch import nn
from ..utils.nn import default_create_nn_layers
from typing import Any, Callable


class bmds(nn.Module):
    def __init__(
            self,
            *args: Any,
            create_nn_layers: Callable[..., Any] = default_create_nn_layers,
            **kwargs: Any,
    ):
        super(bmds, self).__init__()

        layers = create_nn_layers(*args, **kwargs)

        self.compute_variational_parameters = nn.Sequential(*layers)

    def forward(self, dist: Tensor) -> Tensor:
        return self.compute_variational_parameters(dist)
