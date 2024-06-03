import torch


eps = 1e-10


def exponential_log_prob(x, mu, exact=False):
    if exact:
        return -torch.log(mu + eps) - x / (mu + eps)
    else:
        y = x / (mu + eps)
        return 1 - torch.log(y + eps) - y
