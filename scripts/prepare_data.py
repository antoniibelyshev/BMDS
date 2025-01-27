import torch
from torch import Tensor
from torch.utils.data import Subset
from torch_geometric.datasets import TUDataset # type: ignore
from torch_geometric.utils import to_networkx # type: ignore
from torchvision.datasets import MNIST # type: ignore
from torchvision.transforms import ToTensor # type: ignore
import numpy as np
import networkx as nx
from netrd.distance import IpsenMikhailov # type: ignore
from typing import TypeVar, Callable
import os
from itertools import combinations
from tqdm import tqdm


T = TypeVar('T')


def compute_pw_dmat(data: list[T], compute_dist: Callable[[T, T], float]) -> list[list[float]]:
    n = len(data)
    pw_dmat = [[0.0] * n for _ in range(n)]

    for i, j in tqdm(combinations(range(n), 2), total=n * (n - 1) // 2):
        pw_dmat[i][j] = pw_dmat[j][i] = compute_dist(data[i], data[j])

    return pw_dmat


def compute_pw_dmat_vector_data(
        data: Tensor,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tensor:
    n, m = data.size()
    data.to(torch.device(device))
    norm_sqr = data.view(n, 1, m).pow(2).sum(2)
    pw_dist = (norm_sqr - 2 * data @ data.t() + norm_sqr.t()).sqrt()
    return pw_dist.cpu()


def prepare_graph_dataset(
        dataset_name: str,
        compute_dist: Callable[[nx.Graph, nx.Graph], float] = IpsenMikhailov(),
) -> None: 
    dataset = TUDataset(root='tmp_data', name=dataset_name)
    data = [to_networkx(data_point) for data_point in dataset] # type: ignore
    pw_dmat = np.array(compute_pw_dmat(data, compute_dist))
    pw_dmat /= pw_dmat.max()
    labels = dataset.y
    np.savez(f"data/{dataset_name}.npz", pw_dmat=pw_dmat, labels=labels)


def prepare_vector_dataset(dataset_name: str, data: Tensor, labels: Tensor) -> None:
    pw_dmat = compute_pw_dmat_vector_data(data)
    pw_dmat /= pw_dmat.max()
    np.savez(f"data/{dataset_name}.npz", pw_dmat=pw_dmat, labels=labels)


def prepare_mnist_dataset(n_samples: int = 10000) -> None:
    mnist_dataset = Subset(
        MNIST('tmp_data', train=True, download=True, transform=ToTensor()),
        range(n_samples)
    ) # type: ignore
    mnist_data = torch.stack([data.flatten() for data, _ in mnist_dataset], 0)
    labels = torch.tensor([label for _, label in mnist_dataset])
    prepare_vector_dataset("MNIST", mnist_data, labels)


if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")

    prepare_graph_dataset("PROTEINS")
    prepare_mnist_dataset()
