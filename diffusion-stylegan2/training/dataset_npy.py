import os
import numpy as np
import dnnlib
from training.dataset import Dataset

class NPYDataset(Dataset):
    def __init__(self,
        path,
        resolution=None,
        **super_kwargs
    ):
        self._path = path
        self._files = sorted([
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith('.npy')
        ])

        self._raw_idx = np.arange(len(self._files))

        # Infer shape
        sample = np.load(self._files[0])
        assert sample.shape == (256, 256), "Expected (256,256)"

        super().__init__(
            name="npy_dataset",
            raw_shape=[len(self._files), 3, 256, 256],
            **super_kwargs
        )
        
        print(f"[DATASET] Loaded {len(self._files)} samples")
        #self.resolution = 256

    def _load_raw_image(self, raw_idx):
        img = np.load(self._files[raw_idx]).astype(np.float32)

        # garantir [-1,1]
        if img.max() > 1:
            img = img / 127.5 - 1

        # (H,W) -> (3,H,W)
        img = np.stack([img, img, img], axis=0)

        return img

    def _load_raw_labels(self):
        return None