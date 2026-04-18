import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class CTDataset(Dataset):
    def __init__(self, data_root):
        self.paths = sorted(glob.glob(os.path.join(data_root, "*.npy")))

        if len(self.paths) == 0:
            raise ValueError(f"No .npy files found in {data_root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        x = np.load(path)

        x = np.squeeze(x)

        if x.ndim == 3:
            x = x[x.shape[0] // 2]

        if x.shape != (256, 256):
            raise ValueError(f"Wrong shape: {x.shape}")

        x = x.astype(np.float32)

        # normalização robusta
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        x = x * 2 - 1

        # adicionar canal
        x = np.expand_dims(x, axis=0)

        # grayscale -> RGB
        x = np.repeat(x, 3, axis=0)

        x = torch.from_numpy(x)

        return {"image": x}