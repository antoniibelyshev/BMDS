import torch
from torch_ema import ExponentialMovingAverage
import wandb
from typing import Type, Generator, Any, Optional, Union, Dict
from tqdm import trange


def dict_to_device(d: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    for k, v in d.items():
        try:
            d[k] = v.to(device)
        except AttributeError:
            pass
    return d


class BaseTrainer:
    default_optimizer: torch.optim.Optimizer = torch.optim.AdamW
    default_optimizer_kwargs: Dict[str, Any] = {'lr': 2e-4, 'weight_decay': 1e-2}

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        ema_decay: float = 0.999,
        optim_class: Optional[Type[torch.optim.Optimizer]] = None,
        use_ema: bool = True,
        **optimizer_kwargs,
    ):
        self.model = model.to(device)

        self.ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
        self.use_ema = use_ema

        self.device = device

        if optim_class is None:
            optim_class = self.default_optimizer
            optimizer_kwargs = {**self.default_optimizer_kwargs, **optimizer_kwargs}

        self.optimizer = optim_class(
            self.model.parameters(),
            **optimizer_kwargs,
        )

        self.step = 0

    def switch_to_ema(self) -> None:
        self.ema.store(self.model.parameters())
        self.ema.copy_to(self.model.parameters())

    def switch_back_from_ema(self) -> None:
        self.ema.restore(self.model.parameters())

    def calc_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        assert isinstance(batch['x'], torch.Tensor) and isinstance(batch['y'], torch.Tensor)
        return torch.nn.functional.mse_loss(self.model(batch['x']), batch['y'])

    def log_metric(self, metric_name: str, loader_name: str, value: Union[float, torch.Tensor]):
        wandb.log({f'{metric_name}/{loader_name}': value}, step=self.step)

    def optimizer_logic(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.ema.update(self.model.parameters())

    def on_train_iter_start(self, batch: Dict[str, Any]) -> None:
        pass

    def train(
        self,
        train_generator: Generator[Dict[str, Any], None, None],
        total_iters: int = 5000,
        project_name: str = 'default_project',
        experiment_name: str = 'default_experiment',
    ) -> None:
        session = wandb.init(project=project_name, name=experiment_name, entity="antonii-belyshev")

        self.model.train()

        for iter_idx in trange(1, 1 + total_iters):
            self.step = iter_idx

            batch = dict_to_device(next(train_generator), self.device)

            loss = self.calc_loss(batch=batch)
            self.log_metric('loss', 'train', loss)

            self.optimizer_logic(loss)

        self.model.eval()
        if self.use_ema:
            self.switch_to_ema()

        session.finish()


class ClassifierTrainer(BaseTrainer):
    def calc_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        assert isinstance(batch['x'], torch.Tensor) and isinstance(batch['y'], torch.Tensor)
        return torch.nn.functional.cross_entropy(self.model(batch['x']), batch['y'])
