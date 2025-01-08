from typing import TypeVar, Callable
from itertools import combinations
from tqdm import tqdm
import torch
from torch import Tensor


T = TypeVar('T')


def compute_pw_dmat(data: list[T], compute_dist: Callable[[T, T], float]) -> list[list[float]]:
    n = len(data)
    pw_dmat = [[0.0] * n for _ in range(n)]

    for i, j in tqdm(combinations(range(n), 2), total=n * (n - 1) // 2):
        pw_dmat[i][j] = pw_dmat[j][i] = compute_dist(data[i], data[j])

    return pw_dmat


def compute_pw_dmat_vector(
    data: Tensor,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tensor:
    n, m = data.size()
    data.to(torch.device(device))
    norm_sqr = data.view(n, 1, m).pow(2).sum(2)
    pw_dist = norm_sqr - 2 * data @ data.t() + norm_sqr.t()
    return pw_dist.cpu()
