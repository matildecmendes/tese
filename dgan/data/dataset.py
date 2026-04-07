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
        img = np.load(self.files[idx]).astype(np.float32)
        img = np.expand_dims(img, axis=0)  # (1,256,256)
        return torch.from_numpy(img)