import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset

class CTDataset(Dataset):
    def __init__(self, path, max_samples=None):
        self.files = glob.glob(os.path.join(path, "*.npy"))
        if max_samples:
            self.files = self.files[:max_samples]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = np.load(self.files[idx])
        x = torch.tensor(x, dtype=torch.float32)

        # Normalize (IMPORTANT for CT)
        x = (x - x.mean()) / (x.std() + 1e-8)
        x = x.clamp(-3,3)/3

        return x.unsqueeze(0)