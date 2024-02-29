import numpy as np
import torch


from utils import train_eval_k_fold


if __name__ == "__main__":
    N = 10000

    W = torch.tensor(np.load("data/MNIST_W.npy"), dtype=torch.float32)
    # imgs = np.load("data/MNIST_imgs_train.npy")
    labels = torch.tensor(np.load("data/MNIST_labels_train.npy"))

    bmds_kwargs = {"bmds_kwargs": {"max_dim": 100}}
    bmds_train_kwargs = {"max_iter": 3}

    train_eval_k_fold("MNIST", W, labels, k=4, bmds_kwargs=bmds_kwargs, bmds_train_kwargs=bmds_train_kwargs)
