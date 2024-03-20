from typing import Union, Callable
from tqdm import trange
from itertools import combinations


import torch


from .mcmc import langevin_dynamics


DTYPE = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BMDSMCMC:
    samples: torch.Tensor

    def __init__(
            self,
            *,
            m: int = 10,
            init_lr: float = 1e-2,
            gamma: float = -0.8,
            sample_steps: int = 1000,
            sample_size: int = 100,
            n_iter: int = 10000,
            init_log_diff: float = 4.0,
    ):
        self.m = m
        self.init_lr = init_lr
        self.gamma = gamma
        self.sample_steps = sample_steps
        self.sample_size = sample_size
        self.n_iter = n_iter
        self.init_log_diff = init_log_diff

    def create_log_likelihood(
        self,
        dist_mat: torch.Tensor,
        scale: torch.Tensor,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        idx1, idx2 = torch.tensor([*combinations(range(len(dist_mat)), 2)]).T
        dist_sqr = dist_mat[idx1, idx2].pow(2)

        def log_likelihood(x: torch.Tensor) -> torch.Tensor:
            d_sqr = (x[idx1] - x[idx2]).pow(2).sum(dim=1)
            return -torch.log(d_sqr).sum() - (dist_sqr / d_sqr).sum() - (x ** 2 / scale ** 2 / 2).sum() - torch.log(scale).sum() * len(x)
        
        return log_likelihood

    def create_grad_log_likelihood(
        self,
        dist_mat: torch.Tensor,
        scale: torch.Tensor,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        def grad_log_likelihood(x: torch.Tensor) -> torch.Tensor:
            grad = -X / scale ** 2
            for idx1, idx2 in combinations(range(len(dist_mat)), 2):
                diff = x[idx1] - x[idx2]
                d_sqr = diff.pow(2).sum()
                component = 2 * (dist_mat[idx1, idx2].pow(2) - d_sqr) / d_sqr.pow(2) * diff
                grad[idx1] -= component
                grad[idx2] += component
            return grad

    def create_epsilon_scheduler(
            self,
            epoch: int,
    ) -> Callable[[int], float]:
        lr = self.init_lr * self.get_lr(epoch)
        return lambda _: lr

    def get_lr(
            self,
            epoch: int,
    ):
        return self.init_lr * (epoch + 1) ** self.gamma

    def fit(
            self,
            dist_mat_train: torch.Tensor,
    ):
        scale = torch.exp(torch.linspace(0, self.init_log_diff, self.m, device=DEVICE))
        scale = scale / scale.pow(2).sum().pow(0.5)
        samples = torch.randn(self.sample_size, len(dist_mat_train), self.m, device=DEVICE) * scale

        dist_mat_train = (dist_mat_train / dist_mat_train.max()).to(DEVICE)

        vals = []
        for epoch in trange(self.n_iter):
            x = samples[-1]
            scale = samples[-self.sample_size:].pow(2).mean(dim=(0, 1)).pow(0.5)
            log_likelihood = self.create_log_likelihood(dist_mat_train, scale)
            epsilon_scheduler = self.create_epsilon_scheduler(epoch)
            samples, vals_ = langevin_dynamics(x, self.sample_steps, log_likelihood, epsilon_scheduler)
            if torch.isnan(samples).any():
                aaa
            vals.extend(vals_)

        self.samples = samples
        self.vals = vals

    def fit_transform(
            self,
            dist_mat_train: torch.Tensor,
            point_estimate: bool = False
    ):
        self.fit(dist_mat_train)
        return self.samples.mean(dim=0) if point_estimate else self.samples
