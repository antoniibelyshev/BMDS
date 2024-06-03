import torch.nn
from torch import Tensor
from torch import nn
from ..utils.nn import default_create_mlp_layers
from typing import Any, Callable, Dict, Optional, Tuple


class BMDS(nn.Module):
    default_create_layers = default_create_mlp_layers
    default_create_layers_kwargs: Dict[str, Any] = {
        'intermediate_activation': 'PReLU',
        'use_batch_norm': True,
        'last_layer_activation':  True,
        'last_layer_batch_norm': True,
    }

    def __init__(
            self,
            input_dim: int,
            *,
            n_layers: int = 2,
            hidden_dim: int = 1000,
            embedding_dim: int = 100,
            create_layers: Optional[Callable[..., Any]] = None,
            **kwargs: Any,
    ):
        super().__init__()

        if create_layers is None:
            create_layers = self.default_create_layers
            kwargs = {**self.default_create_layers_kwargs, **kwargs}

        head_layers = create_layers(input_dim, [hidden_dim] * (n_layers - 1), hidden_dim, **kwargs)

        self.head = nn.Sequential(*head_layers)
        self.mu = nn.Linear(hidden_dim, embedding_dim)
        self.sigma = nn.Sequential(nn.Linear(hidden_dim, embedding_dim), nn.Softplus())

    def forward(self, dist: Tensor) -> Tuple[Tensor, Tensor]:
        head = self.head(dist)
        return self.mu(head), self.sigma(head)

    def dist(self, dist: Tensor) -> Dict[str, Tensor]:
        mu, sigma = self(dist)
        std = (sigma[..., 0].pow(2) + sigma[..., 1].pow(2)).pow(0.5)
        return {
            'dist': (mu[..., 0] - mu[..., 1] + torch.randn_like(std) * std).pow(2).sum(1).pow(0.5),
            'mu': mu, 'sigma': sigma
        }

    def reg(self, dist: Tensor) -> Dict[str, Tensor]:
        mu, sigma = self(dist)
        lam_sqr = (mu[..., 0].pow(2) + mu[..., 1].pow(2)).mean(0)
        return {'reg': (lam_sqr / sigma).sum(-1).mean(), 'mu': mu, 'sigma': sigma}
