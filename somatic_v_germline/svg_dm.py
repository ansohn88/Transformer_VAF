from typing import Optional

import torch
from numpy import logical_not
from pandas import read_pickle
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split


class TabularDataset(Dataset):
    def __init__(
        self,
        pkl_file: str,
    ) -> None:
        super().__init__()
        self.data = read_pickle(pkl_file)

    def __len__(self):
        return len(self.data["labels"])

    def __getitem__(self, index):
        counts = self.data["counts"][index]
        labels = self.data["labels"][index]

        # accidentally gave somatic a label of 0; invert here:
        # labels = logical_not(labels).astype(int)

        return {
            "counts": torch.from_numpy(counts).float(),
            "labels": torch.from_numpy(labels).long(),
        }


class TabularDataModule(LightningDataModule):
    def __init__(
        self,
        root_file: str,
        batch_size: int = 64,
        shuffle_dataset: bool = True,
        generator: torch.Generator = None,
        num_workers: int = 4,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        self.root_file = root_file
        self.batch_size = batch_size
        self.generator = generator
        self.shuffle_dataset = shuffle_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None):
        dataset_full = TabularDataset(pkl_file=self.root_file)

        test_size = int(len(dataset_full) * 0.15)
        val_size = int(len(dataset_full) * 0.15)
        train_size = int(len(dataset_full) - (test_size + val_size))

        train_ds, val_ds, test_ds = random_split(
            dataset=dataset_full,
            lengths=[train_size, val_size, test_size],
            generator=self.generator,
        )

        if stage == "fit" or stage is None:
            self.train_ds = train_ds
            self.val_ds = val_ds

        if stage == "test":
            self.test_ds = test_ds

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            shuffle=False,
            drop_last=False,
        )
