import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

class VAFDataModule(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()

    def setup(self, stage: str):
        TODO