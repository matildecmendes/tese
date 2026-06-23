import argparse
import os
import random
import shutil
import tempfile
import subprocess
from glob import glob

import torch
from PIL import Image
from torchvision import transforms
from pytorch_msssim import ms_ssim

from pytorch_fid.fid_score import calculate_fid_given_paths
from torch_fidelity import calculate_metrics

# =========================================================
# UTIL FUNCTIONS
# =========================================================

def get_image_paths(folder):
    return sorted(glob(os.path.join(folder, "*.png")))

def log_to_file(output_dir, filename, text):

    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)

    with open(filepath, "a") as f:
        f.write(text + "\n")

def create_temp_subset(paths):

    tmp_dir = tempfile.mkdtemp()

    for i, p in enumerate(paths):
        shutil.copy(p, os.path.join(tmp_dir, f"{i}.png"))

    return tmp_dir

# =========================================================
# MS-SSIM
# =========================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])



def compute_msssim(paths, num_pairs=1000):
    print(f"[INFO] Computing MS-SSIM with {num_pairs} random pairs...")

    scores = []

    for i in range(num_pairs):
        idx1, idx2 = random.sample(range(len(paths)), 2)

        img1 = transform(Image.open(paths[idx1])).unsqueeze(0)
        img2 = transform(Image.open(paths[idx2])).unsqueeze(0)

        score = ms_ssim(img1, img2, data_range=1.0, size_average=True)
        scores.append(score.item())

        if i % 100 == 0:
            print(f"[MS-SSIM] Pair {i}/{num_pairs}")

    return sum(scores) / len(scores)

# =========================================================
# MAE / MSE
# =========================================================

def compute_mae_mse(real_paths, fake_paths, num_pairs=1000):

    print(f"[INFO] Computing MAE/MSE with {num_pairs} random pairs...")

    maes = []
    mses = []

    for i in range(num_pairs):

        real_idx = random.randint(0, len(real_paths) - 1)
        fake_idx = random.randint(0, len(fake_paths) - 1)

        real_img = transform(Image.open(real_paths[real_idx]))
        fake_img = transform(Image.open(fake_paths[fake_idx]))

        mae = torch.mean(torch.abs(real_img - fake_img))
        mse = torch.mean((real_img - fake_img) ** 2)

        maes.append(mae.item())
        mses.append(mse.item())

        if i % 100 == 0:
            print(f"[MAE/MSE] Pair {i}/{num_pairs}")

    mean_mae = sum(maes) / len(maes)
    mean_mse = sum(mses) / len(mses)

    return mean_mae, mean_mse

# =========================================================
# Precision & Recall
# =========================================================

def compute_precision_recall(fake_dir, real_dir):

    metrics = calculate_metrics(
        input1=fake_dir,
        input2=real_dir,
        cuda=True,
        isc=False,
        fid=False,
        kid=False,
        prc=True,
        verbose=False
    )

    precision = metrics["precision"]
    recall = metrics["recall"]

    return precision, recall

# =========================================================
# MAIN
# =========================================================


def main(args):
    random.seed(42)

    fake_paths = get_image_paths(args.fake_dir)
    real_paths = get_image_paths(args.real_dir)

    print(f"[INFO] Fake images: {len(fake_paths)}")
    print(f"[INFO] Real images: {len(real_paths)}")

    assert len(fake_paths) > 0, "No fake images found"
    assert len(real_paths) > 0, "No real images found"

    # =====================================================
    # SELECT RANDOM REAL IMAGES (NO SUBSET SAVED)
    # =====================================================
    n = len(fake_paths)
    print(f"[INFO] Randomly selecting {n} real images (no files saved)...")

    real_selected = random.sample(real_paths, n)

    # =====================================================
    # REAL-vs-REAL FID BASELINE
    # =====================================================
    if args.compute_same_fid:

        if args.fid_dir is None:
            raise ValueError("Need --fid_dir")

        print("\n===== COMPUTING same A-vs-B FID =====")

        fid_paths = get_image_paths(args.fid_dir)

        print(f"[INFO] Total images: {len(fid_paths)}")

        if len(fid_paths) < 2:
            raise ValueError("Need at least 2 images")

        random.seed(42)

        shuffled_paths = fid_paths.copy()
        random.shuffle(shuffled_paths)

        half = len(shuffled_paths) // 2

        subset_A = shuffled_paths[:half]
        subset_B = shuffled_paths[half:half * 2]

        print(f"[INFO] Subset A: {len(subset_A)} images")
        print(f"[INFO] Subset B: {len(subset_B)} images")

        tmp_A = create_temp_subset(subset_A)
        tmp_B = create_temp_subset(subset_B)

        fid = calculate_fid_given_paths(
            [tmp_A, tmp_B],
            batch_size=args.batch_size,
            device='cuda:0',
            dims=2048
        )

        result = (
            f"FID (Subset A vs Subset B): {fid}\n"
            f"Subset A images: {len(subset_A)}\n"
            f"Subset B images: {len(subset_B)}\n"
            f"Random seed: 42"
        )

        print(result)

        log_to_file(
            args.output_dir,
            "fid_a_vs_b.txt",
            result
        )

        shutil.rmtree(tmp_A)
        shutil.rmtree(tmp_B)

        return

    # =====================================================
    # FID
    # =====================================================
    if args.compute_fid:
        print("\n===== COMPUTING FID =====")

        fid = calculate_fid_given_paths(
            [args.fake_dir, args.real_dir],
            batch_size=args.batch_size,
            device='cuda:0',
            dims=2048
        )

        result = f"FID: {fid}"
        print(f"[RESULT] {result}\n")
        log_to_file(args.output_dir, "fid.txt", result)
    # =====================================================
    # CMMD
    # =====================================================
    if args.compute_cmmd:
        print("\n===== COMPUTING CMMD =====")

        cmd = [
            "python",
            os.path.join(args.cmmd_path, "main.py"),
            args.real_dir,
            args.fake_dir,
            f"--batch_size={args.batch_size}",
            f"--max_count={len(fake_paths)}"
        ]

        print("[INFO] Running CMMD command:", " ".join(cmd))

        result = subprocess.run(cmd, check=True, text=True, capture_output=True)

        print(result.stdout)
        log_to_file(args.output_dir, "cmmd.txt", result.stdout)

    # =====================================================
    # MS-SSIM
    # =====================================================
    if args.compute_msssim:
        print("\n===== COMPUTING MS-SSIM =====")

        fake_score = compute_msssim(fake_paths)
        real_score = compute_msssim(real_selected)

        result_fake = f"MS-SSIM (fake): {fake_score}"
        result_real = f"MS-SSIM (real): {real_score}"

        print(f"[RESULT] {result_fake}")
        print(f"[RESULT] {result_real}\n")

        log_to_file(args.output_dir, "msssim.txt", result_fake)
        log_to_file(args.output_dir, "msssim.txt", result_real)

    # =====================================================
    # FRD
    # =====================================================
    if args.compute_frd:
        print("\n===== COMPUTING FRD =====")

        cmd = [
            "python", "-m", "frd_score",
            args.fake_dir,
            args.real_dir,
            "--match_sample_count",
            "--verbose",
            "--num_workers=16",
            "--frd_version", "v0"
        ]

        print("[INFO] Running FRD:", " ".join(cmd))

        result = subprocess.run(cmd, check=True, text=True, capture_output=True)

        print(result.stdout)
        log_to_file(args.output_dir, "frd.txt", result.stdout)

    # =====================================================
    # MAE / MSE
    # =====================================================
    if args.compute_mae_mse:

        print("\n===== COMPUTING MAE / MSE =====")

        mae, mse = compute_mae_mse(
            real_selected,
            fake_paths
        )

        result_mae = f"MAE: {mae}"
        result_mse = f"MSE: {mse}"

        print(f"[RESULT] {result_mae}")
        print(f"[RESULT] {result_mse}\n")

        log_to_file(args.output_dir, "mae.txt", result_mae)
        log_to_file(args.output_dir, "mse.txt", result_mse)

    # =====================================================
    # Precision & Recall
    # =====================================================
    if args.compute_pr:

        print("\n===== COMPUTING PRECISION & RECALL =====")

        real_tmp = create_temp_subset(real_selected)
        fake_tmp = create_temp_subset(fake_paths)

        precision, recall = compute_precision_recall(
            fake_tmp,
            real_tmp
        )

        result_p = f"Precision: {precision}"
        result_r = f"Recall: {recall}"

        print(result_p)
        print(result_r)

        log_to_file(args.output_dir, "precision_recall.txt", result_p)
        log_to_file(args.output_dir, "precision_recall.txt", result_r)

        shutil.rmtree(real_tmp)
        shutil.rmtree(fake_tmp)

# =========================================================
# ARGUMENTS
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--fake_dir', type=str, required=True)
    parser.add_argument('--real_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--fid_dir', type=str, default=None) # calculo fid a vs b

    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--compute_fid', action='store_true')
    parser.add_argument('--compute_cmmd', action='store_true')
    parser.add_argument('--compute_msssim', action='store_true')
    parser.add_argument('--compute_frd', action='store_true')
    parser.add_argument('--compute_mae_mse', action='store_true')
    parser.add_argument('--compute_pr', action='store_true')
    parser.add_argument('--compute_same_fid', action='store_true') # calculo fid a vs b

    parser.add_argument('--cmmd_path', type=str, default='cmmd-pytorch')

    args = parser.parse_args()

    main(args)
