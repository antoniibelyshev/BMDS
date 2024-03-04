from typing import Union, Tuple, List, Dict, Any, Optional

import torch
from pytorch_lightning import LightningModule, Trainer
import numpy as np

from .data import DefaultDataModule
from .utils import check_tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


class MLPClassifier(LightningModule):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: Union[Tuple[int, int], List[int]] = (16, 2),
            act_fn=torch.nn.ReLU,
            optim_class=torch.optim.SGD,
            lr: float = 1e-1,
            loss_fn=torch.nn.CrossEntropyLoss,
    ):
        super().__init__()

        self.input_dim = input_dim
        if isinstance(hidden_dims, Tuple):
            hidden_dims = [hidden_dims[0]] * hidden_dims[1]
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.layers = torch.nn.ModuleList()
        prev_dim = input_dim
        for dim in hidden_dims:
            self.layers.append(torch.nn.Linear(prev_dim, dim))
            self.layers.append(act_fn())
            prev_dim = dim
        self.layers.append(torch.nn.Linear(prev_dim, output_dim))

        self.optim_class = optim_class

        self.lr = lr

        self.loss_fn = loss_fn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        if self.output_dim == 1:
            x = torch.cat((torch.zeros_like(x), x), dim=1)
        return x

    def loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        return self.loss_fn(self(x), y)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        loss = self.loss(batch)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optim_class(self.parameters(), lr=self.lr)


class SklearnMLPClassifier:
    clf: MLPClassifier

    TRAINER_DEFAULTS: Dict[str, Any] = {
        "gpus": 1 if torch.cuda.is_available() else 0,
        "max_epochs": 1000,
    }

    def __init__(
            self,
            n_classes: int,
            *,
            batch_size: Union[int, float] = 1e-2,
            device: torch.device = DEVICE,
            n_avg: int = 1000,
            **clf_kwargs: Any,
    ):
        self.output_dim = n_classes if n_classes > 2 else 1
        self.batch_size = batch_size
        self.device = device
        self.n_avg = n_avg
        self.clf_kwargs = clf_kwargs

    def fit(
            self,
            x_train: Union[torch.Tensor, np.ndarray],
            y_train: Union[torch.Tensor, np.ndarray],
            std_train: Optional[Union[torch.Tensor, np.ndarray]] = None,
            **trainer_kwargs: Any,
    ) -> None:
        x_train = check_tensor(x_train)
        y_train = check_tensor(y_train, dtype=int)
        if std_train is not None:
            std_train = check_tensor(std_train)
        datamodule = DefaultDataModule(
            x_train,
            y_train,
            stds=() if std_train is None else (std_train,),
            batch_size=self.batch_size,
        )
        self.clf = MLPClassifier(x_train.shape[1], self.output_dim, **self.clf_kwargs).to(self.device)
        trainer = Trainer(**{**self.TRAINER_DEFAULTS, **trainer_kwargs})
        trainer.fit(self.clf, datamodule=datamodule)

    def fit_predict(
            self,
            x_train: Union[torch.Tensor, np.ndarray],
            y_train: Union[torch.Tensor, np.ndarray],
            std_train: Optional[Union[torch.Tensor, np.ndarray]] = None,
            **kwargs: Any,
    ) -> torch.Tensor:
        x_train = check_tensor(x_train)
        self.fit(x_train, y_train, std_train=std_train, **kwargs)
        return self.predict(x_train)

    def predict(
            self,
            x_eval: Union[torch.Tensor, np.ndarray],
            std_eval: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> torch.Tensor:
        x_eval = check_tensor(x_eval)
        if std_eval is None:
            return self.clf(x_eval).detach().cpu().argmax(dim=1)
        else:
            std_eval = check_tensor(std_eval)
            x_eval_sample = x_eval + torch.randn(self.n_avg, *x_eval.shape) * std_eval
            return self.clf(x_eval_sample).detach().cpu().mean(dim=0).argmax(dim=1)
