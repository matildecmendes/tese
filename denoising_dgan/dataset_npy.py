import os
import numpy as np
import torch
from torch.utils.data import Dataset


class NPYDataset(Dataset):

    def __init__(
        self,
        folder,
        max_samples=None,
        subset_seed=42
    ):
        # Obter todos os ficheiros NPY numa ordem estável
        files = sorted([
            os.path.join(folder, filename)
            for filename in os.listdir(folder)
            if filename.endswith(".npy")
        ])

        if len(files) == 0:
            raise ValueError(
                f"Não foram encontrados ficheiros .npy em: {folder}"
            )

        print(f"Total de imagens disponíveis: {len(files)}")

        # Escolher aleatoriamente apenas max_samples imagens
        if (
            max_samples is not None
            and max_samples > 0
            and max_samples < len(files)
        ):
            rng = np.random.default_rng(subset_seed)

            selected_indices = rng.choice(
                len(files),
                size=max_samples,
                replace=False
            )

            # Ordenar os índices apenas para manter uma ordem estável
            selected_indices = sorted(selected_indices)

            files = [
                files[index]
                for index in selected_indices
            ]

        self.files = files

        print(f"Imagens utilizadas no treino: {len(self.files)}")
        print(f"Seed usada na seleção: {subset_seed}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        img = np.load(self.files[idx])

        # Garantir float32
        img = img.astype(np.float32)

        # Passar de (256, 256) para (1, 256, 256)
        img = np.expand_dims(img, axis=0)

        # Converter para tensor PyTorch
        img = torch.from_numpy(img)

        return img