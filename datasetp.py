import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset

class CTDataset(Dataset):
    def __init__(self, root):
        self.files = sorted(glob.glob(os.path.join(root, "*.npy")))
        assert len(self.files) > 0, f"No .npy files found in {root}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = np.load(self.files[idx]).astype(np.float32)

        x_min, x_max = x.min(), x.max()
        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)
            x = x * 2.0 - 1.0
        else:
            x = np.zeros_like(x)

        x = torch.from_numpy(x)
        return x.unsqueeze(0)