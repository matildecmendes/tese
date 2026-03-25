import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class CTSliceDataset(Dataset):
    def __init__(self, folder=None, files_list=None, limit=None):
        if files_list is not None:
            self.files = files_list
        else:
            self.files = sorted(os.listdir(folder))
            self.files = [os.path.join(folder, f) for f in self.files]

        if limit:
            self.files = self.files[:limit]

        print(f"[DATASET] Found {len(self.files)} files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        if path.endswith(".npy"):
            img = np.load(path)
        else:
            img = np.array(Image.open(path).convert("L"))

        img = torch.tensor(img).float()

        if img.max() > 1:
            img = img / 127.5 - 1

        if img.ndim == 2:
            img = img.unsqueeze(0)

        return img