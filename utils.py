import pickle
import torch
from configs import config
from torch.utils.data import Dataset


class Dsets(Dataset):
    def __init__(self, mode):
        path = config.dataset_dir + mode + '.pkl'

        with open(path, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    inputs = [a[0] for a in batch]
    target = [int(a[1]) for a in batch]

    maxcLen = max([len(b) for a in inputs for b in a])
    maxwLen = 300

    for sentence in inputs:
        for word in sentence:
            word.extend([0 for _ in range(maxcLen - len(word))])
        sentence.extend([[0 for _ in range(maxcLen)] for _ in range(maxwLen - len(sentence))])

    return torch.tensor(inputs, dtype=torch.int64, device=config.device), \
           torch.tensor(target, dtype=torch.int64, device=config.device)
