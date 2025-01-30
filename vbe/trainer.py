from .vbe import VBE
from .safe_operations import safe_log

import torch
from torch import nn, Tensor
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader
import wandb
from copy import deepcopy
from tqdm import tqdm


def update_ema(model: nn.Module, ema_model: nn.Module, decay: float):
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data = ema_param.data * decay + param.data * (1.0 - decay)


class VBETrainer:
    def __init__(
            self,
            model: VBE,
            ema_model: VBE | None,
            optimizer: Optimizer,
            scheduler: lr_scheduler.LRScheduler | None = None,
            *,
            ema_decay: float = 0.999,
            device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        self.model = model.to(device)
        self.ema_model = ema_model.to(device) if ema_model else deepcopy(model).to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.ema_decay = ema_decay

        self.device = device

    def loss(self, batch: list[Tensor]) -> Tensor:
        x1, x2, s = batch

        dist_sqr = (self.model(x1) - self.model(x2)).square().sum(1)
        ratio = s / dist_sqr
        loss = (ratio - safe_log(ratio) - 1).mean()
        reg = self.model.regularization()
        wandb.log({"reg": reg.item()})
        return loss + reg

    def train(
            self,
            dataloader: DataLoader[tuple[Tensor, ...]],
            epochs: int = 100,
            name: str = 'default',
            project: str | None = 'VBE',
            entity: str | None = None,
    ) -> None:
        run = wandb.init(name=name, project=project, entity=entity)

        with tqdm(total=epochs * len(dataloader)) as pbar:
            for epoch in range(1, epochs + 1):
                self.model.train()

                for batch in dataloader:
                    self.optimizer.zero_grad()
                    loss = self.loss([t.to(self.device) for t in batch])
                    loss.backward()  # type: ignore
                    self.optimizer.step()

                    self.update_ema()

                    wandb.log({"loss": loss.item()})

                    pbar.update()

                    self.eval()

                wandb.log({'epoch': epoch})

                if self.scheduler:
                    self.scheduler.step()

        run.finish()  # type: ignore

        self.model.eval()

    def eval(self) -> None:
        wandb.log({"relevant dims count": self.ema_model.relevant_dims().float().sum()})
        relevance_score = sorted(self.ema_model.relevance_score(), reverse=True)
        for i in range(20):
            wandb.log({f"relevance score {i + 1}": relevance_score[i]})

    def update_ema(self):
        update_ema(self.model, self.ema_model, self.ema_decay)
