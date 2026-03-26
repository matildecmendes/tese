import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 256
CHANNELS = 1
LATENT_DIM = 100

BATCH_SIZE = 8
EPOCHS = 100
LR = 2e-4

DATA_PATH = r"Z:\nas-ctm01\homes\mcmendes\GAN"
SAVE_PATH = "Z:\nas-ctm01\homes\mcmendes\GAN"