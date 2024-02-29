from typing import Union, Callable, TypeVar, Sequence
from itertools import combinations, product
import torch
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

DTYPE = torch.float32
object_type = TypeVar('object_type')


def compute_euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.pow(torch.tensor(x - y, dtype=DTYPE), 2).sum()


def compute_pairwise_dist(
        data1: Sequence[object_type],
        data2: Union[Sequence[object_type], None] = None,
        *,
        eps: float = 1e-6,
        compute_dist: Callable[[object_type, object_type], float] = compute_euclidean_dist,
        n_chunks: Union[int, float] = 10,
) -> torch.Tensor:
    symmetric = data2 is None
    if symmetric:
        data2 = data1
    n = len(data1)
    m = len(data2)
    idx = torch.tensor([*(combinations(range(n), 2) if symmetric else product(range(n), range(m)))])
    dist_mat = torch.zeros(n, m, dtype=DTYPE)
    for idx_chunk in tqdm(torch.chunk(idx, n_chunks, dim=0)):
        with Pool(16) as pool:
            f = partial(compute_dist_, data1=data1, data2=data2, compute_dist=compute_dist)
            dists = torch.tensor(pool.map(f, idx_chunk), dtype=torch.float32)
        dist_mat[tuple(idx_chunk.T)] = dists
        if symmetric:
            dist_mat[tuple(idx_chunk.T)[::-1]] = dists
    return dist_mat + eps * dist_mat.std(dim=None)


def compute_dist_(idx, data1, data2, compute_dist):
    return compute_dist(data1[idx[0]], data2[idx[1]])
