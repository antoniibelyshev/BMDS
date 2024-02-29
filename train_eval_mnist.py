import numpy as np
import torch


from utils import train_eval_k_fold


if __name__ == "__main__":
    N = 10000

    W = torch.tensor(np.load("MNIST_W.npy"), dtype=torch.float32)
    # imgs = np.load("MNIST_imgs_train.npy")
    labels = torch.tensor(np.load("MNIST_labels_train.npy"))

    train_eval_k_fold("MNIST", W, labels)
