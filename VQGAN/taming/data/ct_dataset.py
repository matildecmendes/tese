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

        # -------------------------
        # 1) GARANTIR SHAPE CORRETO
        # -------------------------
        x = np.squeeze(x)

        if x.ndim == 2:
            pass

        elif x.ndim == 3:
            # caso volume (ex: [N,256,256])
            x = x[x.shape[0] // 2]

        else:
            raise ValueError(f"Unsupported shape: {x.shape}")

        if x.shape != (256, 256):
            raise ValueError(f"Final shape wrong: {x.shape}")

        # -------------------------
        # 2) NORMALIZAÇÃO (CRÍTICO)
        # -------------------------
        x = x.astype(np.float32)

        # normalizar para [-1,1] independentemente da escala
        x_min, x_max = x.min(), x.max()

        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)  # [0,1]
            x = x * 2.0 - 1.0                 # [-1,1]
        else:
            x = np.zeros_like(x)

        # -------------------------
        # 3) FORMATO FINAL
        # -------------------------
        x = np.expand_dims(x, axis=0)  # (1,256,256)
        x = torch.from_numpy(x)

        return {"image": x}