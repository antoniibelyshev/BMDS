import io
import zipfile
from typing import Union, Tuple, List

import networkx as nx
import numpy as np
import torch

DTYPE = torch.float32


def check_tensor(t: Union[torch.Tensor, np.ndarray], dtype: torch.dtype = DTYPE) -> torch.Tensor:
    if isinstance(t, torch.Tensor):
        return t
    else:
        return torch.tensor(t, dtype=dtype)


def read_graph_dataset(name: str, path: str = "./data") -> Tuple[List[nx.Graph], np.ndarray]:
    with zipfile.ZipFile(f"{path}/{name}.zip", 'r') as archive:
        with archive.open(f"{name}/{name}_A.txt") as f:
            def handle_line(s: bytes) -> List[str]:
                return s.decode("utf-8").rstrip('\n').split(', ')

            adjacency_list = np.array([*map(handle_line, f.readlines())], dtype=int) - 1
        with archive.open(f"{name}/{name}_graph_indicator.txt") as f:
            graph_indicator = np.loadtxt(io.TextIOWrapper(f), dtype=int) - 1
        with archive.open(f"{name}/{name}_graph_labels.txt") as f:
            graph_labels = np.loadtxt(io.TextIOWrapper(f), dtype=int) - 1

    graphs = get_graphs(adjacency_list, graph_indicator)
    return graphs, graph_labels


def get_graphs(adjacency_list: np.ndarray, graph_indicator: np.ndarray):
    ids, graph_sizes = np.unique(graph_indicator, return_counts=True)
    adjacency_matrices = [np.zeros((graph_size, graph_size)) for graph_size in graph_sizes]
    idx_shifts = np.cumsum(graph_sizes)

    for row, col in adjacency_list:
        graph_id = graph_indicator[row]
        idx_shift = idx_shifts[graph_id - 1] if graph_id else 0
        adjacency_matrices[graph_id][row - idx_shift, col - idx_shift] = 1

    return [nx.Graph(adj) for adj in adjacency_matrices]
