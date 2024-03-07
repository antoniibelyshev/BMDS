from typing import Callable, List

import torch


def langevin_dynamics(
        x_init: torch.Tensor,
        n_iter: int,
        compute_grad: Callable[[torch.Tensor], torch.Tensor],
        epsilon_scheduler: Callable[[int], float],
) -> torch.Tensor:
    samples = [x_init]
    x = x_init
    for k in range(n_iter):
        grad = compute_grad(x)
        epsilon = epsilon_scheduler(k)
        x = langevin_step(x, grad, epsilon)
        samples.append(x)
    return torch.cat(samples, dim=0)


def langevin_step(
        x: torch.Tensor,
        grad: torch.Tensor,
        epsilon: float,
):
    return x + epsilon * grad / 2 + torch.randn_like(x) * epsilon ** 0.5
