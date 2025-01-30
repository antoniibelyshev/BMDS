import torch
from torch import nn, Tensor
from torch.nn.functional import linear
from .safe_operations import safe_log, safe_sqrt, safe_div


class SqueezingLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, eps: float = 5e-2) -> None:
        super().__init__() # type: ignore

        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        
        self.weight = nn.Parameter(Tensor(out_features, in_features))
        self.log_weight_std = nn.Parameter(Tensor(out_features, in_features))

        self.init()

    def init(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=2.23)
        with torch.no_grad():
            self.log_weight_std.copy_(0.5 * self.weight.square().log() - 6)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mu = linear(x, self.weight)
            std = safe_sqrt(linear(x.square(), (2 * self.log_weight_std).exp()))
            return mu + std * torch.randn_like(mu)
        
        return linear(x, self.weight)

    def kl(self) -> Tensor:
        lambda_star_sqr = self.weight.square().mean(1) + self.weight_std_sqr.mean(1)
        return 0.5 * safe_log(lambda_star_sqr).sum() * self.in_features - self.log_weight_std.sum()
    
    def squeeze(self, x: Tensor) -> Tensor:
        return linear(x, self.weight[self.relevant_dims()])        

    def equivalent_dropout_rate(self) -> Tensor:
        alpha = safe_div(self.weight_std_sqr, self.weight.square()).detach().cpu()
        return alpha / (1 + alpha)

    def relevant_dims(self) -> Tensor:
        return (self.equivalent_dropout_rate() < 1 - self.eps).any(1)

    def relevance_score(self) -> Tensor:
        return 1 - self.equivalent_dropout_rate().min(1).values

    @property
    def weight_std_sqr(self):
        return (2 * self.log_weight_std).exp()


class VBE(nn.Module):
    s: Tensor

    def __init__(
        self,
        in_features: int,
        n: int,
        hidden_dim: int = 100,
        encoder_n_layers: int = 1,
        decoder_n_layers: int = 2,
        d: int = 100,
    ) -> None:
        super(VBE, self).__init__() # type: ignore

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

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(self.squeezing_layer(self.encoder(x)))

    def embed(self, x: Tensor) -> Tensor:
        return self.squeezing_layer.squeeze(self.encoder(x)).detach().cpu()

    def regularization(self) -> Tensor:
        return 2 * self.squeezing_layer.kl() / (self.n * (self.n - 1))

    def relevant_dims(self) -> Tensor:
        return self.squeezing_layer.relevant_dims()

    def relevance_score(self) -> Tensor:
        return self.squeezing_layer.relevance_score()

    @property
    def device(self):
        return next(self.parameters()).device
