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

        # remover dimensões inúteis
        x = np.squeeze(x)

        #  garantir que é 2D
        if x.ndim != 2:
            raise ValueError(f"Unexpected shape after squeeze: {x.shape}")

        # adicionar canal
        x = x[None, ...]  # (1,256,256)

        x = torch.tensor(x, dtype=torch.float32)

        print("RAW SHAPE:", x.shape)

        return {"image": x}