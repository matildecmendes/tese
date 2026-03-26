import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CTDataset(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.files = []

        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(".npy"):
                    self.files.append(os.path.join(root, f))

        print(f"Found {len(self.files)} npy files")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.load(self.files[idx])  # shape esperado: (H, W)

        # DEBUG shape
        if idx == 0:
            print("Original shape:", img.shape, img.min(), img.max())

        # converter para tensor
        img = torch.tensor(img, dtype=torch.float32)

        # garantir shape (1, H, W)
        if len(img.shape) == 2:
            img = img.unsqueeze(0)

        # resize
        img = self.transform(img)

        # NORMALIZAÇÃO PARA [-1, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = img * 2 - 1

        return img