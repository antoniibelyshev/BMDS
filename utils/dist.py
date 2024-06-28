import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from scipy.integrate import quad
import networkx as nx
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial
from itertools import combinations


def compute_sqrt_eigv(graph):
    return np.sqrt(np.abs(eigh(laplacian(nx.to_numpy_array(graph), normed=False))[0][1:]))


def compute_sqrt_eigvals(graphs, cpus=None, chunksize=1):
    with Pool(cpus) as p:
        return [*p.imap(compute_sqrt_eigv, tqdm(graphs), chunksize)]


def _im(w1, w2, hwhm=0.08):
    N = len(w1) + 1

    norm1 = (N - 1) * np.pi / 2 - np.sum(np.arctan(-w1 / hwhm))
    norm2 = (N - 1) * np.pi / 2 - np.sum(np.arctan(-w2 / hwhm))

    # define both spectral densities
    density1 = lambda w: np.sum(hwhm / ((w - w1) ** 2 + hwhm**2)) / norm1
    density2 = lambda w: np.sum(hwhm / ((w - w2) ** 2 + hwhm**2)) / norm2

    func = lambda w: (density1(w) - density2(w)) ** 2

    return np.sqrt(quad(func, 0, np.inf, limit=100)[0])


def compute_im(w):
    return _im(*w)


def fast_compute_pw_im(w, cpus=None):
    with Pool(cpus) as p:
        return [*p.imap(compute_im, tqdm(combinations(w, 2), total=len(w) * (len(w) - 1) // 2))]


def compute_pw_im(graphs, cpus=None):
    # cpus = cpus or cpu_count()
    print(f"Using {cpus} cpus")
    print('Computing laplacians\' spectrums')
    # w = compute_sqrt_eigvals(graphs, cpus, chunksize)
    w = list(map(compute_sqrt_eigv, tqdm(graphs)))
    print('Computing pairwise distances')
    
    idx = np.array([*combinations(range(len(w)), 2)])
    idx1, idx2 = idx.T

    dist_mat = np.zeros((len(w), len(w)))
    dist_mat[idx1, idx2] = dist_mat[idx2, idx1] = fast_compute_pw_im(w, cpus)

    return dist_mat
