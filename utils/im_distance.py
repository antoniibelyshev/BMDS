import networkx as nx
from scipy.stats import gaussian_kde
from scipy.integrate import quad
from tqdm import tqdm
import numpy as np
from itertools import combinations


def normalized_laplacian_spectrum(graph: nx.Graph):
    L = nx.laplacian_matrix(graph).todense()
    eigenvalues = np.linalg.eigvalsh(L)
    return eigenvalues


def spectral_density(w: np.ndarray, points: np.ndarray, hwhm: float):
    return (hwhm / ((points - w[:, None]) ** 2 + hwhm ** 2)).sum(axis=0)


def im_dist(d1: np.ndarray, d2: np.ndarray, dx: float):
    norm1 = d1.sum() * dx
    norm2 = d2.sum() * dx + (len(d1) - len(d2)) * np.pi / 2
    return np.sqrt(((d1 / norm1 - d2 / norm2) ** 2).sum() * dx)


def compute_pw_im_distance(graphs: list[nx.Graph], num_points: int = 10000, hwhm: float = 0.08):
    n = len(graphs)
    pw_dmat = np.zeros((n, n))

    w_lst = [np.sqrt(np.abs(nx.laplacian_spectrum(g)[1:])) for g in graphs]
    spectral_range = np.linspace(0, 10 * max(map(max, w_lst)), num_points)
    dx = spectral_range[1] - spectral_range[0]
    d_lst = [(hwhm / ((spectral_range - w[:, None]) ** 2 + hwhm ** 2)).sum(axis=0) for w in tqdm(w_lst)]

    for i, j in tqdm(combinations(range(n), 2), total=n * (n - 1) // 2):
        d1, d2 = d_lst[i], d_lst[j]
        norm1 = np.sqrt((d1 ** 2).sum() * dx)
        norm2 = np.sqrt((d2 ** 2).sum() * dx)
        norm1 = d1.sum() * dx
        norm2 = d2.sum() * dx + (len(w_lst[i]) - len(w_lst[j])) * np.pi / 2
        norm1 = 2 * len(w_lst[i])
        norm2 = len(w_lst[i]) + len(w_lst[j])
        pw_dmat[i][j] = pw_dmat[j][i] = np.sqrt(((d1 / norm1 - d2 / norm2) ** 2).sum() * dx)

    return pw_dmat


# def compute_pw_im_cuda(graphs: list[nx.Graph]):
#     eigvals = [np.linalg.eigvalsh(nx.laplacian_matrix(g)) for g in graphs]

