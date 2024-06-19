import torch.nn
from torch import Tensor
from torch import nn
from .nn import create_mlp_layers
from .distributions import exponential_log_prob
from .trainer import BaseTrainer
from typing import Any, Callable, Optional, Dict, List, Tuple


eps = 1e-10


class BMDS(nn.Module):
    default_create_layers_kwargs: Dict[str, Any] = {
        'activation': 'PReLU',
        'use_batch_norm': False,
        'last_layer_activation':  True,
        'last_layer_batch_norm': True,
        'allow_sparse_input': True,
    }

    def __init__(
            self,
            input_dim: int,
            n: int,
            *,
            n_layers: int = 2,
            hidden_dim: int = 1000,
            embedding_dim: int = 100,
            create_layers: Optional[Callable[..., List[nn.Module]]] = None,
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
        self.sigma = nn.Parameter(torch.randn(hidden_dim, embedding_dim) / hidden_dim * 1e-6)

    def forward(self, inp: Tensor) -> Tuple[Tensor, Tensor]:
        head = self.head(inp)
        mu = head @ self.mu
        sigma = (head.pow(2) @ self.sigma.pow(2)).pow(0.5)
        return mu, sigma

    def compute_log_prob(self, true_dist, mu, sigma) -> dict:
        dist = sample_dist(mu, sigma)
        return exponential_log_prob(true_dist.pow(2), dist.pow(2)).mean()

    def compute_reg(self, mu, sigma):
        return (torch.log(self.mu.pow(2).mean(0) + self.sigma.pow(2).mean(0)).sum() * self.hidden_dim -
                torch.log(self.sigma.pow(2)).sum()) / self.n / 2

    def additional_metrics(self, mu, sigma) -> Dict[str, Tensor]:
        scales = list(sorted(rms(mu, (0, 1))))
        idx = list(range(5)) + [9, 14, 19]
        return {f"scale #{i}": scales[-i] for i in idx}

    def loss(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # assert isinstance(batch['dist'], Tensor) and isinstance(batch['inp'], Tensor)
        mu, sigma = self(batch['inp'])
        if len(mu.shape) == 2:
            mu = mu.reshape(-1, 2, mu.shape[1])
            sigma = sigma.reshape(-1, 2, sigma.shape[1])
        log_prob = self.compute_log_prob(batch['dist'], mu, sigma)
        # reg = (self.log_lambda - self.log_sigma + (mu.pow(2).mean((0, 1)) + torch.exp(2 * self.log_sigma)) /
        #        torch.exp(2 * self.log_lambda) / 2).sum() / (self.n - 1) * 2 - 1
        # reg = (torch.log((mu.pow(2)).mean((0, 1)) + torch.exp(2 * self.log_sigma) + eps) - 2 * self.log_sigma).sum() / (self.n - 1)
        reg = self.compute_reg(mu, sigma)
        return {'loss': -log_prob + reg, 'log_prob': log_prob, 'reg': reg, **self.additional_metrics(mu, sigma)}


class SparseBMDS(BMDS):
    def __init__(self, dist_matrix, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dist_matrix = dist_matrix


    def loss(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        assert isinstance(batch['idx'], Tensor)
        idx = batch['idx']
        true_dist = torch.tensor(self.dist_matrix[list(idx.T[0]), list(idx.T[1])])
        
        mu, sigma = self.forward(csr_to_tensor(self.dist_matrix[idx.reshape(-1)]))
        mu = mu.reshape(*idx.shape, -1)
        sigma = sigma.reshape(*idx.shape, -1)
        
        log_prob = self.compute_log_prob(true_dist, mu, sigma)
        reg = self.compute_reg(mu, sigma)

        return {'loss': -log_prob + reg, 'log_prob': log_prob, 'reg': reg, **self.additional_metrics(mu, sigma)}


def csr_to_tensor(scipy_csr):
    data = scipy_csr.data
    indices = scipy_csr.indices
    indptr = scipy_csr.indptr

    # Convert these components to PyTorch tensors
    values = torch.tensor(data, dtype=torch.float32)
    col_indices = torch.tensor(indices, dtype=torch.int64)
    crow_indices = torch.tensor(indptr, dtype=torch.int64)

    # Create a sparse CSR tensor in PyTorch
    return torch.sparse_csr_tensor(crow_indices, col_indices, values, size=scipy_csr.shape)


def sample_dist(mu: Tensor, sigma: Tensor) -> Tensor:
    std = rms(sigma, 1, False)
    return rms(mu[:, 0] - mu[:, 1] + torch.randn_like(std) * std, -1)


def rms(x, dim, avg=True):
    return (torch.mean if avg else torch.sum)(x.pow(2), dim).pow(0.5)


class BMDSTrainer(BaseTrainer):
    def __init__(self, bmds: BMDS, *args, **kwargs):
        super().__init__(bmds, *args, **kwargs)

    def calc_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        loss = self.model.loss(batch)
        for name, val in loss.items():
            if name != 'loss':
                self.log_metric(name, 'train', val)
        return loss['loss']
