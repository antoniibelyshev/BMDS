from typing import Union, Callable
from tqdm import trange


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
            gamma: float = -0.8,
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

    @staticmethod
    def create_grad_log_likelihood(
            scale: torch.Tensor,
            dist_sqr: torch.Tensor,
            neighbors_idx: torch.Tensor,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        def grad_log_likelihood(x):
            neighbors = x[neighbors_idx]
            diffs = x - neighbors
            d_sqr = diffs.pow(2).sum(dim=2, keepdims=True)
            return ((dist_sqr - d_sqr) / d_sqr.pow(2) * diffs).mean(dim=0) - x / scale.pow(2)
        return grad_log_likelihood

    def create_epsilon_scheduler(
            self,
            epoch: int,
    ) -> Callable[[int], float]:
        lr = self.get_lr(epoch)
        return lambda _: lr

    def get_lr(
            self,
            epoch: int,
    ):
        return (epoch + 1) ** self.gamma

    def fit(
            self,
            dist_mat_train: torch.Tensor,
    ):
        scale = torch.linspace(0, self.init_log_diff, self.m, device=DEVICE)
        scale = scale / scale.pow(2).sum().pow(0.5)
        samples = torch.randn(len(dist_mat_train), self.sample_size, self.m, device=DEVICE) * scale

        dist_mat_train = (dist_mat_train / dist_mat_train.max()).to(DEVICE)
        if isinstance(self.n_neighbors, float):
            self.n_neighbors = int(self.n_neighbors * len(samples))
        neighbors_idx = dist_mat_train.argsort(dim=0)[1:self.n_neighbors + 1]

        dist_sqr = dist_mat_train[torch.arange(len(samples))][neighbors_idx].pow(2).unsqueeze(2)

        for epoch in trange(self.n_iter):
            x = samples[-1]
            scale = samples[-self.sample_size:].pow(2).mean(dim=(0, 1))
            grad_log_likelihood = self.create_grad_log_likelihood(scale, dist_sqr, neighbors_idx)
            epsilon_scheduler = self.create_epsilon_scheduler(epoch)
            samples = langevin_dynamics(x, self.sample_steps, grad_log_likelihood, epsilon_scheduler)

        self.samples = samples

    def fit_transform(
            self,
            dist_mat_train: torch.Tensor,
            point_estimate: bool = False
    ):
        self.fit(dist_mat_train)
        return self.samples.mean(dim=0) if point_estimate else self.samples
