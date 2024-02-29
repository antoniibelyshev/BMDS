import numpy as np
import torch


from utils import train_eval_k_fold


if __name__ == "__main__":
    N = 10000

    W = torch.tensor(np.load("data/MNIST_W.npy"), dtype=torch.float32)
    # imgs = np.load("data/MNIST_imgs_train.npy")
    labels = torch.tensor(np.load("data/MNIST_labels_train.npy"))

    train_eval_k_fold("MNIST", W, labels, k=4)
