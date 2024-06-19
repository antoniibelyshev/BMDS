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
        'use_batch_norm': False,
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

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n = n

        if create_layers is None:
            create_layers = create_mlp_layers
            kwargs = {**self.default_create_layers_kwargs, **kwargs}

        head_layers = create_layers(input_dim, [hidden_dim] * (n_layers - 1), hidden_dim, **kwargs)

        self.head = nn.Sequential(*head_layers)

        self.mu = nn.Parameter(torch.randn(hidden_dim, embedding_dim) / hidden_dim)
        self.sigma = nn.Parameter(torch.randn(hidden_dim, embedding_dim) / hidden_dim)

    def forward(self, inp: Tensor) -> tuple[Tensor, Tensor]:
        head = self.head(inp.reshape(-1, self.input_dim).pow(2))
        mu = head @ self.mu
        sigma = (head.pow(2) @ self.sigma.pow(2)).pow(0.5)
        return mu.reshape(*inp.shape[:-1], -1), sigma.reshape(*inp.shape[:-1], -1)

    def loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        assert isinstance(batch['dist'], Tensor) and isinstance(batch['inp'], Tensor)
        mu, sigma = self(batch['inp'])
        dist = sample_dist(mu, sigma)
        log_prob = exponential_log_prob(batch['dist'].pow(2), dist.pow(2)).mean()
        # reg = (self.log_lambda - self.log_sigma + (mu.pow(2).mean((0, 1)) + torch.exp(2 * self.log_sigma)) /
        #        torch.exp(2 * self.log_lambda) / 2).sum() / (self.n - 1) * 2 - 1
        # reg = (torch.log((mu.pow(2)).mean((0, 1)) + torch.exp(2 * self.log_sigma) + eps) - 2 * self.log_sigma).sum() / (self.n - 1)
        reg = (torch.log(self.mu.pow(2).mean(0) + self.sigma.pow(2).mean(0)).sum() * self.hidden_dim -
               torch.log(self.sigma.pow(2)).sum()) / self.n / (self.n - 1)
        return {'loss': -log_prob + reg, 'log_prob': log_prob, 'reg': reg,
                **{f"scale #{i}": scale for i, scale in enumerate(sorted(mu.pow(2).mean((0, 1)).pow(0.5))[-1:-6:-1])},
                **{f"loglam #{i}": loglam for i, loglam in enumerate(sorted(self.log_lambda)[-1:-6:-1])}}


def sample_dist(mu: Tensor, sigma: Tensor) -> Tensor:
    std = sigma.pow(2).sum(dim=-2).pow(0.5)
    return (mu[..., 0, :] - mu[..., 1, :] + torch.randn_like(std) * std).pow(2).mean(-1).pow(0.5)


class BMDSTrainer(BaseTrainer):
    def __init__(self, bmds: BMDS, *args, **kwargs):
        super().__init__(bmds, *args, **kwargs)

    def calc_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        loss = self.model.loss(batch)
        for name, val in loss.items():
            if name != 'loss':
                self.log_metric(name, 'train', val)
        return loss['loss']
