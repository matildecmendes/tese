import os
import numpy as np
from training.dataset import Dataset

class NPYDataset(Dataset):
    def __init__(self, path, resolution=None, **super_kwargs):
        self._path = path

        self._files = sorted([
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".npy")
        ])

        assert len(self._files) > 0, f"No npy files found in {path}"

        sample = np.load(self._files[0])

        if sample.ndim == 2:
            sample = sample[None, :, :]  # (1,H,W)

        raw_shape = [len(self._files)] + list(sample.shape)

        super().__init__(
            name="npy_dataset",
            raw_shape=raw_shape,
            **super_kwargs
        )

    def _load_raw_image(self, raw_idx):
        x = np.load(self._files[raw_idx])

        if x.ndim == 2:
            x = x[None, :, :]

        x = x.astype(np.float32)

        # DEBUG 
        if raw_idx == 0:
            print("DEBUG DATASET SAMPLE:")
            print("shape:", x.shape)
            print("min:", x.min(), "max:", x.max())

        return x

    def _load_raw_labels(self):
        return None

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])

        if self._xflip[idx]:
            image = image[:, :, ::-1]

        return image.copy(), self.get_label(idx)