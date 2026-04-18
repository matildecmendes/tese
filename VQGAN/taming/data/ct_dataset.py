import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CTDataset(Dataset):
    def __init__(self, data_root, file_list):
        with open(file_list, "r") as f:
            self.paths = f.read().splitlines()

        self.data_root = data_root

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = os.path.join(self.data_root, self.paths[idx])
        x = np.load(path)  # shape (256,256)

        # garantir formato [C,H,W]
        if x.ndim == 2:
            x = x[None, ...]

        x = torch.tensor(x, dtype=torch.float32)

        return x