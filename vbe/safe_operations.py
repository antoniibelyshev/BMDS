from torch import Tensor


eps = 1e-10


def safe_log(x: Tensor) -> Tensor:
    return (x + eps).log()


def safe_sqrt(x: Tensor) -> Tensor:
    return (x + eps).sqrt()


def safe_div(x: Tensor, y: Tensor) -> Tensor:
    return x / (y + eps)
