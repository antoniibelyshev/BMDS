from typing import Union, Tuple, List, Dict, Any

import torch
from pytorch_lightning import LightningModule, Trainer

from .data import DefaultDataModule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLPClassifier(LightningModule):
    def __init__(
            self,
            input_dim: int,
            n_classes: int,
            hidden_dims: Union[Tuple[int, int], List[int]] = (16, 2),
            act_fn=torch.nn.ReLU,
            optim_class=torch.optim.SGD,
            lr: float = 1e-2,
            loss_fn=torch.nn.CrossEntropyLoss,
    ):
        super().__init__()

        self.input_dim = input_dim
        if isinstance(hidden_dims, Tuple):
            hidden_dims = [hidden_dims[0]] * hidden_dims[1]
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes

        self.layers = torch.nn.ModuleList()
        prev_dim = input_dim
        for dim in hidden_dims:
            self.layers.append(torch.nn.Linear(prev_dim, dim))
            self.layers.append(act_fn())
            prev_dim = dim
        self.layers.append(torch.nn.Linear(prev_dim, n_classes - (n_classes == 2)))

        self.optim_class = optim_class

        self.lr = lr

        self.loss_fn = loss_fn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        if self.n_classes == 2:
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
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.device = device
        self.n_avg = n_avg
        self.clf_kwargs = clf_kwargs

    def fit(
            self,
            x_train: torch.Tensor,
            y_train: torch.Tensor,
            std_train: Union[torch.Tensor, None] = None,
            **trainer_kwargs: Any,
    ) -> None:
        datamodule = DefaultDataModule(
            x_train,
            y_train,
            stds=() if std_train is None else (std_train,),
            batch_size=self.batch_size,
        )
        self.clf = MLPClassifier(x_train.shape[1], self.n_classes, **self.clf_kwargs).to(self.device)
        trainer = Trainer(**{**self.TRAINER_DEFAULTS, **trainer_kwargs})
        trainer.fit(self.clf, datamodule=datamodule)

    def fit_predict(
            self,
            x_train: torch.Tensor,
            y_train: torch.Tensor,
            std_train: Union[torch.Tensor, None] = None,
            **kwargs: Any,
    ) -> torch.Tensor:
        self.fit(x_train, y_train, std_train=std_train, **kwargs)
        return self.predict(x_train)

    def predict(
            self,
            x_eval: torch.Tensor,
            std_eval: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        if std_eval is None:
            return self.clf(x_eval).detach().cpu().argmax(dim=1)
        else:
            return self.clf(x_eval + torch.randn(self.n_avg, *x_eval.shape)).detach().cpu().mean(dim=0).argmax(dim=1)
