from typing import Optional

from lightning import LightningDataModule
from sklearn import model_selection
from torch.utils.data import DataLoader

from dataset import VAFDataset


class VAFDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        k: int,
        num_splits: int = 5,
        batch_size: int = 1,
        num_workers: int = 8,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # self.data_dir = data_dir
        # self.num_splits = num_splits
        # self.batch_size = batch_size
        # self.num_workers = num_workers
        # self.pin_memory = pin_memory

        assert (
            1 <= self.hparams.k + 1 <= self.hparams.num_splits
        ), "incorrect fold number"

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    def setup(self, stage=None):
        if not self.data_train and not self.data_val:
            dataset_full = VAFDataset(data_dir=self.hparams.data_dir)

            kf = model_selection.KFold(
                n_splits=self.hparams.num_splits, shuffle=True, random_state=42
            )
            all_splits = [k for k in kf.split(dataset_full)]

            train_indexes, val_indexes = all_splits[self.hparams.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            self.data_train, self.data_val = (
                # dataset_full[train_indexes],
                # dataset_full[val_indexes],
                [dataset_full[i] for i in train_indexes],
                [dataset_full[i] for i in val_indexes],
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
