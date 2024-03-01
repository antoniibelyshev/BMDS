from abc import abstractmethod
from typing import Tuple, Any, Union, Callable, Dict, TypeVar, Sequence, List


import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT

from .data import DefaultDataModule, DistMatrixDataModule


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


class BMDS(LightningModule):
    x: torch.Tensor
    std: torch.Tensor
    dim: int

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
            idx: Union[int, None] = None,
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
        self.log("dim", float(self.dim), prog_bar=True)
        del batch, loss, reg
        return total_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optim([self.x, self.std], self.lr)


class BMDSTrain(BMDS):
    def __init__(
            self,
            n: int,
            *,
            max_dim: int = 10,
            lr: float = 1e-3,
            n_iters_check: int = 100,
            threshold: float = 0.0,
            optim: Callable[[List[torch.Tensor], float], torch.optim.Optimizer] = torch.optim.Adam,
            device: torch.device = DEVICE,
    ):
        super().__init__(n, max_dim, lr=lr, optim=optim, device=device)

        self.max_dim = max_dim
        self.n_iters_check = n_iters_check
        self.threshold = threshold

        self.scale = self.compute_scale()

        self.dim = max_dim
        self.loss_diffs: List[float] = []

    def sample_dist_sqr(
            self,
            idx: torch.Tensor,
    ) -> torch.Tensor:
        diffs = self.x[idx.T[0], :self.dim] - self.x[idx.T[1], :self.dim]
        std = (self.std[idx.T[0], :self.dim] ** 2 + self.std[idx.T[1], :self.dim] ** 2) ** 0.5
        xi = torch.randn_like(diffs) * std
        dist_sqr = ((diffs + xi) ** 2).sum(axis=-1)
        del idx, diffs, std, xi
        return dist_sqr

    def compute_scale(self) -> torch.Tensor:
        return (self.x ** 2).mean(dim=0) + (self.std ** 2).mean(dim=0)

    def regularization(
            self,
            idx: Union[int, None] = None,
    ) -> torch.Tensor:
        self.scale = self.compute_scale()
        l, r = (0, self.dim) if idx is None else (idx, idx + 1)
        return (torch.log(self.scale[l:r]).sum() - torch.log(self.std[:, l:r] ** 2).mean(dim=0).sum()) / (self.n - 1)

    def on_train_batch_start(
            self,
            batch: Any,
            batch_idx: int
    ) -> None:
        with torch.no_grad():
            u, s, v = torch.svd(self.x)
            orthogonal_x = u * s
            orthogonal_x[:, self.dim:] = 0
            self.x.copy_(orthogonal_x)
            del u, s, v, orthogonal_x

    def on_train_batch_end(
            self,
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int
    ) -> None:
        with torch.no_grad():
            self.dim -= 1
            loss_ = self.loss(batch)
            loss_diff = loss_ - outputs["loss"]
            reg_diff = -self.regularization(self.dim)
            self.loss_diffs.append(loss_diff + reg_diff)
            if not self.keep_change():
                self.dim += 1
            else:
                self.loss_diffs = []

        del batch, outputs, loss_, loss_diff, reg_diff

    def keep_change(self) -> bool:
        if len(self.loss_diffs) < self.n_iters_check:
            return False
        else:
            diff_samples = torch.tensor(self.loss_diffs[-self.n_iters_check:])
            mu = diff_samples.mean()
            std = diff_samples.std(dim=None)
            res = mu / std < -self.threshold
            self.log("frac", mu / std, prog_bar=True)
            del diff_samples, mu, std
            return res


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

        self.alpha = ((x_train ** 2).mean(dim=0) + (std_train ** 2).mean(axis=0)).to(device)

    def sample_dist_sqr(
            self,
            idx: torch.Tensor,
    ) -> torch.Tensor:
        diffs = self.x - self.x_train[idx, None]
        std = (self.std ** 2 + self.std_train[idx, None] ** 2) ** 0.5
        xi = torch.randn_like(diffs) * std
        dist_sqr = ((diffs + xi) ** 2).sum(dim=2)
        del idx, diffs, std, xi
        return dist_sqr

    def regularization(
            self,
            idx: Union[int, None] = None
    ) -> torch.Tensor:
        return (self.alpha * (self.x ** 2 + self.std ** 2) - torch.log(self.alpha * self.std ** 2)).sum() / 2 / self.m


object_type = TypeVar('object_type')


class SklearnBMDS:
    dim: int
    x_train: torch.Tensor
    std_train: torch.Tensor

    TRAINER_DEFAULTS: Dict[str, Any] = {
        "gpus": 1 if torch.cuda.is_available() else 0,
        "max_epochs": 1000,
    }

    def __init__(
            self,
            *,
            batch_size_train: int = 1000,
            batch_size_eval: int = 1e-2,
            bmds_train_kwargs: Union[Dict[str, Any], None] = None,
            bmds_eval_kwargs: Union[Dict[str, Any], None] = None,
            device: torch.device = DEVICE,
    ):
        self.batch_size_train = batch_size_train
        self.batch_size_eval = batch_size_eval
        self.bmds_train_kwargs: Dict[str, Any] = bmds_train_kwargs or dict()
        self.bmds_eval_kwargs: Dict[str, Any] = bmds_eval_kwargs or dict()
        self.device = device

    def fit(
            self,
            dist_mat_train: torch.Tensor,
            **trainer_kwargs: Any,
    ) -> None:
        datamodule = DistMatrixDataModule(dist_mat_train, batch_size=self.batch_size_train)
        bmds_train = BMDSTrain(
            len(dist_mat_train),
            device=DEVICE,
            **self.bmds_train_kwargs,
        )

        print("\nLearning the optimal train embedding...")
        trainer = Trainer(**{**self.TRAINER_DEFAULTS, **trainer_kwargs})
        trainer.fit(bmds_train, datamodule=datamodule)

        self.dim = bmds_train.dim
        self.x_train = bmds_train.x[:, :self.dim].detach().cpu()
        self.std_train = bmds_train.std[:, :self.dim].detach().cpu()

    def fit_transform(
            self,
            train_data: Union[Sequence[object_type], torch.Tensor],
            **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.fit(train_data, **kwargs)
        return self.x_train, self.std_train

    def transform(
            self,
            dist_mat_eval: torch.Tensor,
            **trainer_kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        datamodule = DefaultDataModule(
            torch.arange(len(dist_mat_eval)),
            dist_mat_eval,
            batch_size=self.batch_size_eval,
        )
        bmds_eval = BMDSEval(
            dist_mat_eval.shape[1],
            self.x_train,
            self.std_train,
            device=DEVICE,
            **self.bmds_eval_kwargs,
        )

        print("\nLearning the optimal train embedding...")
        trainer = Trainer(**{**self.TRAINER_DEFAULTS, **trainer_kwargs})
        trainer.fit(bmds_eval, datamodule=datamodule)

        return bmds_eval.x.detach().cpu(), bmds_eval.std.detach().cpu()
