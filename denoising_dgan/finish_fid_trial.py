import os
from pytorch_fid.fid_score import calculate_fid_given_paths
import torch

TRIAL_DIR = "/nas-ctm01/homes/mcmendes/outputs/random_search/trial_002"

REAL_IMG_DIR = "/nas-ctm01/datasets/public/lidc-mcmendes/slices_png"

FID_SAMPLE_DIR = os.path.join(TRIAL_DIR, "fid_samples")

print("Counting generated images...")

n_imgs = len([
    f for f in os.listdir(FID_SAMPLE_DIR)
    if f.endswith(".png")
])

print(f"Found {n_imgs} generated images")

if n_imgs == 0:
    raise RuntimeError(
        "No generated images found in fid_samples/"
    )

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Computing FID...")

fid = calculate_fid_given_paths(
    [FID_SAMPLE_DIR, REAL_IMG_DIR],
    batch_size=50,
    device=device,
    dims=2048
)

print(f"FID = {fid:.4f}")

fid_file = os.path.join(TRIAL_DIR, "fid.txt")

with open(fid_file, "w") as f:
    f.write(str(fid))

print(f"Saved FID to {fid_file}")