from abc import abstractmethod
from typing import Tuple, Any, Callable, Dict, TypeVar, List, Optional, Union


import torch
from pytorch_lightning import LightningModule, Trainer
import numpy as np

from .data import DefaultDataModule, DistMatrixDataModule
from .utils import check_tensor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


class BMDS(LightningModule):
    def __init__(
            self,
            n: int,
            dim: int,
            *,
            lr: float,
            optim: Callable[[List[torch.Tensor], float], torch.optim.Optimizer] = torch.optim.Adam,
            device: torch.device = DEVICE,
    ):
        super().__init__()

        sigma = 1 / (dim + 1e-6) ** 0.5
        self.x = torch.randn(n, dim, device=device) * sigma
        self.x.requires_grad = True
        self.std = torch.randn(n, dim, device=device) * sigma
        self.std.requires_grad = True
        self.dim = dim

        self.n = n
        self.lr = lr
        self.optim = optim

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
        return self.optim([self.x, self.std], self.lr)


class BMDSTrain(BMDS):
    mask: torch.Tensor

    def __init__(
            self,
            n: int,
            *,
            max_dim: int = 10,
            lr: float = 1e-3,
            threshold: float = 0.1,
            optim: Callable[[List[torch.Tensor], float], torch.optim.Optimizer] = torch.optim.Adam,
            device: torch.device = DEVICE,
    ):
        super().__init__(n, max_dim, lr=lr, optim=optim, device=device)

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


class BMDSEval(BMDS):
    def __init__(
            self,
            n: int,
            x_train: torch.Tensor,
            std_train: torch.Tensor,
            *,
            device: torch.device = DEVICE,
            optim: Callable[[List[torch.Tensor], float], torch.optim.Optimizer] = torch.optim.Adam,
            lr: float = 1e-3,
    ):
        super().__init__(n, x_train.shape[1], lr=lr, optim=optim, device=device)

        self.x_train = x_train.to(device)
        self.std_train = std_train.to(device)
        self.m = len(x_train)

        self.lam_sqr = (x_train.pow(2).mean(dim=0) + std_train.pow(2).mean(axis=0)).to(device)

    def sample_dist_sqr(
            self,
            idx: torch.Tensor,
    ) -> torch.Tensor:
        diffs = self.x - self.x_train[idx, None]
        std = (self.std.pow(2) + self.std_train[idx, None].pow(2)).pow(0.5)
        xi = torch.randn_like(diffs) * std
        dist_sqr = (diffs + xi).pow(2).sum(dim=2)
        del idx, diffs, std, xi
        return dist_sqr

    def regularization(
            self,
            idx: Optional[int] = None
    ) -> torch.Tensor:
        return ((self.x.pow(2) + self.std.pow(2)) / self.lam_sqr -
                torch.log(self.std.pow(2) / self.lam_sqr)).sum() / 2 / self.m


object_type = TypeVar('object_type')


class SklearnBMDS:
    dim: int
    x_train: torch.Tensor
    std_train: torch.Tensor
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
        datamodule = DistMatrixDataModule(dist_mat_train, batch_size=self.batch_size_train)
        bmds_train = BMDSTrain(
            len(dist_mat_train),
            device=DEVICE,
            **self.bmds_train_kwargs,
        )

        print("\nLearning the optimal train embedding...")
        trainer = Trainer(**{**self.TRAINER_DEFAULTS, **trainer_kwargs})
        trainer.fit(bmds_train, datamodule=datamodule)

        self.x_train = bmds_train.x.detach().cpu()
        self.std_train = bmds_train.std.detach().cpu()

        self.max_dist = dist_mat_train.max()

    def fit_transform(
            self,
            dist_mat_train: torch.Tensor,
            **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.fit(dist_mat_train, **kwargs)
        return self.x_train, self.std_train

    def transform(
            self,
            dist_mat_eval: torch.Tensor,
            **trainer_kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist_mat_eval = check_tensor(dist_mat_eval)
        datamodule = DefaultDataModule(
            torch.arange(len(dist_mat_eval)),
            (dist_mat_eval / self.max_dist) ** 2,
            batch_size=self.batch_size_eval,
        )
        bmds_eval = BMDSEval(
            dist_mat_eval.shape[1],
            self.x_train,
            self.std_train,
            device=DEVICE,
            **self.bmds_eval_kwargs,
        )

        print("\nLearning the optimal eval embedding...")
        trainer = Trainer(**{**self.TRAINER_DEFAULTS, **trainer_kwargs})
        trainer.fit(bmds_eval, datamodule=datamodule)

        return bmds_eval.x.detach().cpu(), bmds_eval.std.detach().cpu()
