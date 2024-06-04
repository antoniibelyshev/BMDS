import torch.nn
from torch import Tensor
from torch import nn
from .nn import create_mlp_layers
from .distributions import exponential_log_prob
from .trainer import BaseTrainer
from typing import Any, Callable, Optional


eps = 1e-10


class BMDS(nn.Module):
    default_create_layers_kwargs: dict[str, Any] = {
        'activation': 'PReLU',
        'use_batch_norm': True,
        'last_layer_activation':  True,
        'last_layer_batch_norm': True,
    }

    def __init__(
            self,
            input_dim: int,
            n: int,
            *,
            n_layers: int = 2,
            hidden_dim: int = 1000,
            embedding_dim: int = 100,
            create_layers: Optional[Callable[..., list[nn.Module]]] = None,
            **kwargs: Any,
    ):
        super().__init__()

        self.n = n

        if create_layers is None:
            create_layers = create_mlp_layers
            kwargs = {**self.default_create_layers_kwargs, **kwargs}

        head_layers = create_layers(input_dim, [hidden_dim] * (n_layers - 1), hidden_dim, **kwargs)

        self.head = nn.Sequential(*head_layers)
        self.mu = nn.Linear(hidden_dim, embedding_dim)
        self.log_sigma = nn.Linear(hidden_dim, embedding_dim)

        self.log_lam = nn.Parameter(torch.ones(embedding_dim) / embedding_dim, requires_grad=True)

    def forward(self, dist: Tensor) -> tuple[Tensor, Tensor]:
        head = self.head(dist)
        return self.mu(head), self.log_sigma(head)

    def loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        assert isinstance(batch['dist'], Tensor) and isinstance(batch['inp'], Tensor)
        mu, log_sigma = self(batch['inp'])
        dist = self.sample_dist(mu, log_sigma)
        log_prob = exponential_log_prob(batch['dist'], dist).mean()
        reg = (self.log_lambda - log_sigma.mean((0, 1)) + (mu.pow(2) + torch.exp(2 * log_sigma)).mean((0, 1)) /
               torch.exp(2 * self.log_lam)).sum() / self.n / (self.n - 1) * 2 - 1
        return {'loss': -log_prob + reg, 'log_prob': log_prob, 'reg': reg}


def sample_dist(mu: Tensor, log_sigma: Tensor) -> dict[str, Tensor]:
    std = torch.exp(2 * log_sigma).sum(-1).pow(0.5)
    return (mu[..., 0, :] - mu[..., 1, :] + torch.randn_like(std) * std).pow(2).sum(-1).pow(0.5)


class BMDSTrainer(BaseTrainer):
    def __init__(self, bmds: BMDS, *args, **kwargs):
        super().__init__(bmds, *args, **kwargs)

    def calc_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        loss = self.model.loss(batch)
        for name, val in loss.items():
            if name != 'loss':
                self.log_metric(name, 'train', val)
        return loss['loss']
