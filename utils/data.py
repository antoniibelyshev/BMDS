from torch.utils.data import Dataset
from typing import TypeVar


K = TypeVar('K')
T = TypeVar('T')


class NamedDataset(Dataset):
    def __init__(self, names: list[K], data: list[T]):
        self.names = names
        self.data = data

    def __len__(self) -> int:
        return len(self.data[0])

    def __getitem__(self, idx) -> dict[K, T]:
        return {name: data[idx] for name, data in zip(self.names, self.data)}
