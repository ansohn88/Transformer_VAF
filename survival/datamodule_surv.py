from typing import Optional

import torch
from pandas import read_pickle
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split


class TabularDataset(Dataset):
    def __init__(self,
                 pkl_file: str,
                 time_bins_file: str
                 ) -> None:
        super().__init__()
        self.data = read_pickle(pkl_file)
        self.time_bin_tgts = read_pickle(time_bins_file)

    def __len__(self):
        return len(self.data['tcga_id'])

    def __getitem__(self, index):
        tcga_id = self.data['tcga_id'][index]
        counts = self.data['counts'][index]
        gs_lbl = self.data['is_somatic'][index]
        event = self.data['event'][index]
        os = self.data['OS.time'][index]
        dss = self.data['DSS.time'][index]
        dfi = self.data['DFI.time'][index]
        pfi = self.data['PFI.time'][index]
        # purity = self.data['purity'][index]
        # ploidy = self.data['ploidy'][index]

        time_bin_tgt = self.time_bin_tgts[tcga_id]

        discrete_lbl = time_bin_tgt['discrete_lbl']
        time_bins = time_bin_tgt['time_bins']
        cancer_type = time_bin_tgt['type']
        eval_times = time_bin_tgt['eval_times']

        return {
            'tcga_id': tcga_id,
            'counts': torch.from_numpy(counts).float(),
            'gs_lbl': torch.from_numpy(gs_lbl).long(),
            'event': torch.tensor(event).float(),
            'os': torch.from_numpy(os).float(),
            'dss': torch.from_numpy(dss).float(),
            'dfi': torch.from_numpy(dfi).float(),
            'pfi': torch.from_numpy(pfi).float(),
            # 'purity': torch.from_numpy(purity).float(),
            # 'ploidy': torch.from_numpy(ploidy).float(),
            'discrete_lbl': torch.tensor([discrete_lbl]).long(),
            'time_bins': torch.from_numpy(time_bins).float(),
            'eval_times': torch.from_numpy(eval_times).long(),
            'cancer_type': cancer_type
        }


class TabularDataModule(LightningDataModule):
    def __init__(
        self,
        root_file: str,
        time_bins_file: str,
        batch_size: int = 64,
        shuffle_dataset: bool = True,
        generator: torch.Generator = None,
        num_workers: int = 4,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        self.root_file = root_file
        self.time_bins_file = time_bins_file
        self.batch_size = batch_size
        self.generator = generator
        self.shuffle_dataset = shuffle_dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None):
        dataset_full = TabularDataset(
            pkl_file=self.root_file,
            time_bins_file=self.time_bins_file
        )

        test_size = int(len(dataset_full) * 0.15)
        val_size = int(len(dataset_full) * 0.15)
        train_size = int(len(dataset_full) - (test_size + val_size))

        train_ds, val_ds, test_ds = random_split(
            dataset=dataset_full,
            lengths=[train_size,
                     val_size,
                     test_size],
            generator=self.generator
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
