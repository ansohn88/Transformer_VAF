import pandas as pd
from torch.utils.data import Dataset


class VAFDataset(Dataset):
    def __init__(self, data_dir, is_train, bsize):
        super().__init__()

        self.is_train = is_train

        self.bsize = bsize

        # (
        #     self.sample_id,
        #     self.counts,
        #     self.events,
        #     self.time_to_events,
        # ) = self.prepare_dataset(data_dir)

        self.data = self.prepare_dataset(data_dir)

    def prepare_dataset(self, data_dir):
        out = pd.read_pickle(data_dir)

        sample_id = list(out["id"])
        counts = list(out["counts"])
        event = list(out["event"])
        time_to_event = list(out["time"])

        data = []
        for i, (s, c, e, t) in enumerate(zip(sample_id, counts, event, time_to_event)):
            data.append([s, c, e, t])

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.is_train:
            if index == len(self.data) - 1:
                # index_b = np.random.randint(len(self.data))
                index_b = np.random.randint(
                    low=0, high=int(len(self.data) - 1), size=int(self.bsize - 1)
                )
            else:
                # index_b = np.random.randint(index + 1, len(self.data))
                index_b = np.random.randint(
                    low=index + 1, high=len(self.data), size=int(self.bsize - 1)
                )
            print(f"SOME #s: {index}--{index_b}")
            # return [
            #     [self.data[index][i], self.data[index_b][i]]
            #     for i in range(len(self.data[index]))
            # ]
            batched_data = [[self.data[index][i]] for i in range(len(self.data[index]))]
            for b in range(self.bsize - 1):
                for i in range(len(self.data[index])):
                    batched_data[i].append(self.data[index_b.item(b)][i])
            return batched_data
        else:
            return self.data[index]
