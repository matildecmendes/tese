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

        # DEBUG (mantém isto temporariamente!)
        print("RAW SHAPE:", x.shape)

        x = np.squeeze(x)

        # CASOS CORRETOS
        if x.ndim == 2:
            pass

        # CASO 3D (ex: volume ou canal extra)
        elif x.ndim == 3:
            # escolher a dimensão menor como canal
            if 1 in x.shape:
                x = np.squeeze(x)
            else:
                # volume 3D → escolher slice central
                x = x[x.shape[0] // 2]

        else:
            raise ValueError(f"Unsupported shape: {x.shape}")

        # garantir final
        if x.shape != (256, 256):
            raise ValueError(f"Final shape wrong: {x.shape}")

        x = x[None, ...]  # (1,256,256)

        print("FINAL SHAPE:", x.shape)

        x = torch.tensor(x, dtype=torch.float32)

        return {"image": x}