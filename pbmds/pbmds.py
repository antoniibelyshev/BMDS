import torch
from torch import nn, Tensor
from torch.nn.functional import linear
from utils import safe_log, safe_sqrt


class SqueezingLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, threshold: float = 5e-2) -> None:
        super().__init__() # type: ignore

        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        
        self.weight = nn.Parameter(Tensor(out_features, in_features))
        self.weight_std = nn.Parameter(Tensor(out_features, in_features))

        self.init()

    def init(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=2.23)
        with torch.no_grad():
            self.weight_std.copy_(self.weight * 0.1)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mu = linear(x, self.weight)
            std = safe_sqrt(linear(x.square(), self.weight_std.square()))

            return mu + std * torch.randn_like(mu)
        
        return linear(x, self.weight)

    def kl(self) -> Tensor:
        sigma_sqr = self.weight.square().mean(1) + self.weight_std.square().mean(1)
        return 0.5 * (safe_log(sigma_sqr).sum() * self.in_features - safe_log(self.weight_std.square()).sum())
    
    def squeeze(self, x: Tensor) -> Tensor:
        return linear(x, self.weight[self.relevant_dims()])        

    def equivalent_dropout_rate(self) -> Tensor:
        alpha = (self.weight / self.weight_std).square()
        return alpha / (1 + alpha)

    def relevant_dims(self) -> tuple[Tensor, Tensor]:
        return (self.equivalent_dropout_rate() > self.threshold).any(1) # type: ignore


class PBMDS(nn.Module):
    s: Tensor

    def __init__(
        self,
        in_features: int,
        n: int,
        hidden_dim: int = 100,
        encoder_n_layers: int = 2,
        decoder_n_layers: int = 2,
        d: int = 100,
    ) -> None:
        super(PBMDS, self).__init__() # type: ignore

        self.n = n

        self.encoder = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                *[
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
                    for _ in range(encoder_n_layers - 1)
                ]
            )

        self.squeezing_layer = SqueezingLinear(hidden_dim, hidden_dim)

        self.decoder = nn.Sequential(
                *[
                    nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
                    for _ in range(decoder_n_layers - 1)
                ],
                nn.Linear(hidden_dim, d)
            )

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)
    
    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        z = self.encode(x)
        z = self.squeezing_layer(z)
        return self.decode(z)

    def embedding(self, x: Tensor) -> Tensor:
        encoding = self.encode(x)
        return self.squeezing_layer.squeeze(encoding).detach().cpu()

    def regularization(
            self,
    ) -> Tensor:
        return 2 * self.squeezing_layer.kl() / self.n / (self.n - 1)

    def relevant_dims(self) -> tuple[Tensor, Tensor]:
        return self.squeezing_layer.relevant_dims()
