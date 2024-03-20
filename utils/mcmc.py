from typing import Callable, List, Optional

import torch
from typing import Tuple, List


def langevin_dynamics(
        x_init: torch.Tensor,
        n_iter: int,
        log_likelihood: Callable[[torch.Tensor], torch.Tensor],
        epsilon_scheduler: Callable[[int], float],
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    vals = []
    samples = [x_init]
    x = x_init
    for k in range(n_iter):
        x.requires_grad = True
        val = log_likelihood(x)
        grad = torch.autograd.grad(val, x)[0]
        x.requires_grad = False
        vals.append(val.detach())
        epsilon = epsilon_scheduler(k)
        x = langevin_step(x, grad, epsilon)
        samples.append(x)
    return torch.stack(samples, dim=0), vals


def langevin_step(
        x: torch.Tensor,
        grad: torch.Tensor,
        epsilon: float,
):
    return x + epsilon * grad / 2 + torch.randn_like(x) * epsilon ** 0.5
