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

        print("✅ USING CORRECT CT DATASET")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = np.load(self.paths[idx])

        # (256,256)
        if x.shape != (256, 256):
            raise ValueError(f"Unexpected shape: {x.shape}")

        # [-1,1]
        if x.min() < -1.1 or x.max() > 1.1:
            raise ValueError(f"Data not in [-1,1]: min={x.min()}, max={x.max()}")

        # adicionar canal → (1,256,256)
        x = np.expand_dims(x, axis=0)

        # converter para tensor
        x = torch.from_numpy(x).float()

        # garantir contiguous (importante para conv2d)
        x = x.contiguous()

        return x