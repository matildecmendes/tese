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

        # DEBUG 
        # print("shape original:", x.shape)

        # garantir formato correto
        if x.ndim == 2:
            x = x[None, ...]

        elif x.ndim == 3:
            # casos comuns errados
            if x.shape[0] == 256:
                x = x.transpose(1, 0, 2)  # (256,1,256) -> (1,256,256)

            elif x.shape[-1] == 1:
                x = x.transpose(2, 0, 1)  # (256,256,1) -> (1,256,256)

        x = torch.tensor(x, dtype=torch.float32)

        return {"image": x}