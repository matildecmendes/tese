import os
import numpy as np
import torch
from torch.utils.data import Dataset

class NPYDataset(Dataset):
    def __init__(self, folder):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.load(self.files[idx])  # shape: (256,256)

        # garantir float32
        img = img.astype(np.float32)

        # adicionar canal -> (1,256,256)
        img = np.expand_dims(img, axis=0)

        # converter para tensor
        img = torch.from_numpy(img)

        # garantir [-1,1]
        if img.max() > 1.0:
            img = img / 127.5 - 1.0

        return img