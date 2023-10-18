import pandas as pd
from torch.utils.data import Dataset


class VAFDataset(Dataset):
    def __init__(self, file_path: str) -> None:
        super().__init__()

        self.file_path = file_path

        self.barcodes, self.counts, self.events, self.os_months = self.prepare_data(
            file_path=file_path
        )

    @staticmethod
    def prepare_data(file_path: str) -> tuple:
        data_dict = pd.read_pickle(file_path)

        barcodes = data_dict["id"]
        counts = data_dict["counts"]
        events = data_dict["events"]
        os_months = data_dict["time"]

        return barcodes, counts, events, os_months

    def __len__(self):
        return len(self.barcodes)

    def __getitem__(self, index):
        sample_barcode = self.barcodes[index]
        variates = self.counts[index]
        event = self.events[index]
        time_to_event = self.os_months[index]

        sample = {
            "sample_barcode": sample_barcode,
            "variates": variates,
            "event": event,
            "time_to_event": time_to_event,
        }

        return sample
