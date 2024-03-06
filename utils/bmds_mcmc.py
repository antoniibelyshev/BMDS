from typing import Union
from tqdm import trange


import torch


DTYPE = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BMDSMCMC:
    scale: torch.Tensor
    x: torch.Tensor

    def __init__(
            self,
            *,
            m: int = 10,
            gamma: float = 0.8,
            sample_steps: int = 1000,
            sample_size: int = 100,
            n_iter: int = 10000,
            init_log_diff: float = 4.0,
            n_neighbors: Union[int, float] = 0.2,
    ):
        self.m = m
        self.gamma = gamma
        self.sample_steps = sample_steps
        self.sample_size = sample_size
        self.n_iter = n_iter
        self.init_log_diff = init_log_diff
        self.n_neighbors = n_neighbors

    def run_mcmc(
            self,
            x_init: torch.Tensor,
            neighbors: torch.Tensor,
            dist_sqr: torch.Tensor,
    ) -> torch.Tensor:
        samples = [x_init.clone()]
        for k in range(self.sample_steps):
            grad = self.grad_log_likelihood(samples[-1], neighbors, dist_sqr)
            samples.append(self.next_x(samples[-1], grad, (k + 1) ** self.gamma))
        return torch.tensor(samples[-self.sample_size:], dtype=DTYPE)

    def grad_log_likelihood(
            self,
            x: torch.Tensor,
            neighbors: torch.Tensor,
            dist_sqr: torch.Tensor,
    ) -> torch.Tensor:
        diffs = x - neighbors
        d_sqr = diffs.pow(2).sum(dim=2, keepdims=True)
        return ((dist_sqr - d_sqr) / d_sqr.pow(2) * diffs).sum(dim=0).mean(dim=1) - x / self.scale.pow(2)

    @staticmethod
    def next_x(
            x: torch.Tensor,
            grad: torch.Tensor,
            epsilon: float,
    ) -> torch.Tensor:
        return x + epsilon * grad / 2 + torch.randn_like(x) * epsilon ** 0.5

    def fit(
            self,
            dist_mat_train: torch.Tensor,
    ):
        scale = torch.linspace(0, self.init_log_diff, self.m, device=DEVICE)
        self.scale = scale / scale.pow(2).sum().pow(0.5)
        self.x = torch.randn(len(dist_mat_train), self.sample_size, self.m, device=DEVICE) * self.scale

        dist_mat_train = (dist_mat_train / dist_mat_train.max()).to(DEVICE)
        if isinstance(self.n_neighbors, float):
            self.n_neighbors = int(self.n_neighbors * len(self.x))
        neighbors_idx = dist_mat_train.argsort(dim=1)[:, 1:self.n_neighbors + 1]

        for _ in trange(self.n_iter):
            idx = torch.randint(0, len(self.x), size=(1,), device=DEVICE)
            neighbors = self.x[neighbors_idx[idx]]
            dist_sqr = dist_mat_train[idx][neighbors_idx[idx]].unsqueeze(1).pow(2)
            self.x[idx] = self.run_mcmc(self.x[idx, -1], neighbors, dist_sqr)
            self.scale = self.x.pow(2).sum(dim=0).mean(dim=1).pow(0.5)

    def fit_transform(
            self,
            dist_mat_train: torch.Tensor,
    ):
        self.fit(dist_mat_train)
        return self.x
