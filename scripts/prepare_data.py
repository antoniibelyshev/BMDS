from utils import compute_pw_dmat, compute_pw_dmat_vector_data, compute_pw_im_distance

import torch
from torch import Tensor
from torch.utils.data import TensorDataset, Subset
from torch_geometric.datasets import TUDataset # type: ignore
from torch_geometric.utils import to_networkx # type: ignore
from torchvision.datasets import MNIST # type: ignore
from torchvision.transforms import ToTensor # type: ignore
import numpy as np
import networkx as nx
from netrd.distance import IpsenMikhailov # type: ignore
from typing import Callable


def compute_im_dist(g1: nx.Graph, g2: nx.Graph) -> float:
    return IpsenMikhailov()(g1, g2)


def prepare_graph_dataset(
    dataset_name: str,
    compute_dist: Callable[[nx.Graph, nx.Graph], float] = compute_im_dist,
) -> None: 
    dataset = TUDataset(root='tmp_data', name=dataset_name)
    data = [to_networkx(data_point) for data_point in dataset]
    # pw_dmat = np.array(compute_pw_im_distance(data))
    pw_dmat = np.array(compute_pw_dmat(data, compute_dist))
    pw_dmat /= pw_dmat.max()
    labels = dataset.y
    np.savez(f"data/{dataset_name}.npz", pw_dmat=pw_dmat, labels=labels)


def prepare_vector_dataset(dataset_name: str, data: Tensor, labels: Tensor) -> None:
    pw_dmat = compute_pw_dmat_vector_data(data)
    pw_dmat /= pw_dmat.max()
    np.savez(f"data/{dataset_name}.npz", pw_dmat=pw_dmat, labels=labels)


def prepare_mnist_dataset(n_samples: int = 10000) -> None:
    mnist_dataset: TensorDataset = Subset(
        MNIST('tmp_data', train=True, download=True, transform=ToTensor()),
        range(n_samples)
    ) # type: ignore
    mnist_data = torch.stack([data.flatten() for data, _ in mnist_dataset], 0)
    labels = torch.tensor([label for _, label in mnist_dataset])
    prepare_vector_dataset("MNIST", mnist_data, labels)


if __name__ == "__main__":
    prepare_graph_dataset("PROTEINS")
    prepare_graph_dataset("IMDB-BINARY")
    prepare_mnist_dataset()
