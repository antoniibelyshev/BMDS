from abc import abstractmethod
from typing import Tuple, Any, Callable, Dict, List, Optional, Union, Iterable


import torch
from pytorch_lightning import LightningModule, Trainer
import numpy as np

from .data import DistMatrixDataModule
from .utils import check_tensor, gen_mlp


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


class BMDS(LightningModule):
    x: torch.Tensor
    std: torch.Tensor

    def __init__(
            self,
            s: torch.Tensor,
            dim: int,
            *,
            lr: float,
            optim: Callable[[Iterable[torch.Tensor], float], torch.optim.Optimizer] = torch.optim.Adam,
            gen_network: Callable[[int, int], torch.Tensor] = gen_mlp,
            device: torch.device = DEVICE,
    ):
        super().__init__()

        self.s = s.to(device)
        self.dim = dim

        self.n = len(s)
        self.lr = lr
        self.optim = optim

        self.compute_parameters = gen_network(self.n, 2 * dim).to(device)

    @abstractmethod
    def sample_dist_sqr(
            self,
            idx: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def loss(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        idx, observed_dist_sqr = batch
        sampled_dist_sqr = self.sample_dist_sqr(idx)
        ratio = observed_dist_sqr / sampled_dist_sqr
        loss = (ratio - torch.log(ratio) - 1).mean(dim=0).sum()
        del batch, idx, observed_dist_sqr, sampled_dist_sqr, ratio
        return loss

    @abstractmethod
    def regularization(
            self,
            idx: Optional[int] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        parameters = self.compute_parameters(self.s).reshape(-1, self.dim, 2)
        self.x = parameters[..., 0]
        self.std = parameters[..., 1]

    def training_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        loss = self.loss(batch)
        self.log("loss", loss)
        reg = self.regularization()
        self.log("reg", reg)
        total_loss = loss + reg
        self.log("total_loss", total_loss)
        del batch, loss, reg
        return total_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optim(self.parameters(), self.lr)


class BMDSTrain(BMDS):
    mask: torch.Tensor

    def __init__(
            self,
            s: torch.Tensor,
            *,
            max_dim: int = 10,
            lr: float = 1e-3,
            threshold: float = 0.1,
            optim: Callable[[Iterable[torch.Tensor], float], torch.optim.Optimizer] = torch.optim.Adam,
            device: torch.device = DEVICE,
    ):
        super().__init__(s, max_dim, lr=lr, optim=optim, device=device)

        self.max_dim = max_dim
        self.threshold = threshold

        self.dim = max_dim
        self.total_loss_diffs: List[float] = []
        self.total_losses: List[float] = []

    def sample_dist_sqr(
            self,
            idx: torch.Tensor,
    ) -> torch.Tensor:
        diffs = self.x[idx.T[0]] - self.x[idx.T[1]]
        std = (self.std[idx.T[0]].pow(2) + self.std[idx.T[1]].pow(2)).pow(0.5)
        xi = torch.randn_like(diffs) * std
        dist_sqr = (diffs + xi).pow(2).sum(axis=-1)
        del idx, diffs, std, xi
        return dist_sqr

    def regularization(
            self,
            idx: Optional[int] = None,
    ) -> torch.Tensor:
        scale = self.x.pow(2).mean(dim=0) + self.std.pow(2).mean(dim=0)
        reg = (torch.log(scale).sum() - torch.log(self.std.pow(2)).mean(dim=0).sum()) / (self.n - 1)
        del scale
        return reg


class SklearnBMDS:
    dim: int
    bmds_train: BMDSTrain
    max_dist: float

    TRAINER_DEFAULTS: Dict[str, Any] = {
        "gpus": 1 if torch.cuda.is_available() else 0,
        "max_epochs": 500,
    }

    def __init__(
            self,
            *,
            batch_size_train: int = 5e-4,
            batch_size_eval: int = 1e-2,
            bmds_train_kwargs: Optional[Dict[str, Any]] = None,
            bmds_eval_kwargs: Optional[Dict[str, Any]] = None,
            device: torch.device = DEVICE,
    ):
        self.batch_size_train = batch_size_train
        self.batch_size_eval = batch_size_eval
        self.bmds_train_kwargs: Dict[str, Any] = bmds_train_kwargs or dict()
        self.bmds_eval_kwargs: Dict[str, Any] = bmds_eval_kwargs or dict()
        self.device = device

    def fit(
            self,
            dist_mat_train: Union[torch.Tensor, np.ndarray],
            **trainer_kwargs: Any,
    ) -> None:
        dist_mat_train = check_tensor(dist_mat_train)
        self.max_dist = dist_mat_train.max()
        datamodule = DistMatrixDataModule(dist_mat_train, batch_size=self.batch_size_train)
        self.bmds_train = BMDSTrain(
            dist_mat_train.pow(2),
            device=DEVICE,
            **self.bmds_train_kwargs,
        )

        print("\nLearning the optimal train embedding...")
        trainer = Trainer(**{**self.TRAINER_DEFAULTS, **trainer_kwargs})
        trainer.fit(self.bmds_train, datamodule=datamodule)

    def fit_transform(
            self,
            dist_mat_train: torch.Tensor,
            **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.fit(dist_mat_train, **kwargs)
        return self.transform(dist_mat_train)

    def transform(
            self,
            dist_mat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist_mat = check_tensor(dist_mat) / self.max_dist

        parameters = self.bmds_train.compute_parameters(dist_mat).reshape(-1, self.dim, 2)
        x = parameters[..., 0]
        std = parameters[..., 1]

        return x, std
