from torch import nn, Tensor


class NNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        *,
        hidden_dims: list[int] = [100] * 2,
        act_fn: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        self.clf = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            act_fn(),
            *[
                nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i + 1]), act_fn())
                for i in range(len(hidden_dims) - 1)
            ],
            nn.Linear(hidden_dims[-1], n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.clf(x)
