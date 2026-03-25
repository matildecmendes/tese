import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class CTDataset(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.paths = sorted(glob.glob(os.path.join(root_dir, "*.png")))
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]

        img = Image.open(img_path).convert("L")  # grayscale
        img = img.resize((self.image_size, self.image_size))

        img = np.array(img).astype(np.float32)

        # NORMALIZAÇÃO IGUAL À TUA
        img = (img / 127.5) - 1.0

        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)

        return img