import torch
from torch import nn, Tensor
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader
import wandb
from typing import Any, TypeVar, Generic
from copy import deepcopy


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eps = 1e-10


T = TypeVar('T', bound=nn.Module)


def update_ema(model: nn.Module, ema_model: nn.Module, decay: float):
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data = ema_param.data * decay + param.data * (1.0 - decay)


class BaseTrainer(Generic[T]):
    def __init__(
        self,
        model: T,
        ema_model: T | None,
        optimizer: Optimizer,
        scheduler: lr_scheduler.LRScheduler | None = None,
        *,
        ema_decay: float = 0.999,
        device: torch.device = DEVICE,
    ):
        self.model = model.to(device)
        self.ema_model = ema_model.to(device) if ema_model else deepcopy(model).to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.ema_decay = ema_decay

        self.device = device

    def loss(self, batch: list[Tensor]) -> Tensor:
        raise NotImplementedError
    
    def train(
        self,
        dataloader: DataLoader[tuple[Tensor, ...]],
        epochs: int = 100,
        name: str = 'default',
        project: str = 'PBMDS',
        entity: str = "ai-prentice",
        **eval_kwargs: Any,
    ) -> None:        
        run = wandb.init(name=name, project=project, entity=entity)

        for epoch in range(1, epochs + 1):
            self.model.train()

            for batch in dataloader:
                self.optimizer.zero_grad()
                loss = self.loss((t.to(self.device) for t in batch))
                loss.backward() # type: ignore
                self.optimizer.step()

                self.update_ema()

                wandb.log({"loss": loss.item()})

            self.eval(**eval_kwargs)

            wandb.log({'epoch': epoch})

            if self.scheduler:
                self.scheduler.step()

        run.finish() # type: ignore

        self.model.eval()

    def eval(self, **kwargs: Any) -> Any:
        pass
    
    def update_ema(self):
        update_ema(self.model, self.ema_model, self.ema_decay)
