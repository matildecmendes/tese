import os
import cv2
import numpy as np
import torch
import argparse

from residualUnet import *
from metrics import dice_coef, loss_dice
from segmentation_functions import *

from keras.optimizers import Adam


# =====================================================
# ARGUMENTS
# =====================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_dir",
    type=str,
    required=True,
    help="Folder containing PNG images"
)

parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Folder where masks and overlays will be saved"
)

parser.add_argument(
    "--weights",
    type=str,
    required=True,
    help="Path to bestmodel.epoch05.hdf5"
)

parser.add_argument(
    "--num_images",
    type=int,
    default=10,
    help="Number of images to process"
)

args = parser.parse_args()

INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# LOAD MODEL
# =====================================================

print("Loading model...")

resUnet = residualUNet()

opt = Adam(learning_rate=1e-4)

resUnet.compile(
    optimizer=opt,
    loss=loss_dice,
    metrics=[dice_coef]
)

resUnet.load_weights(args.weights)

print("Model loaded.")


# =====================================================
# IMAGE LIST
# =====================================================

files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(".png")
])[:args.num_images]

print(f"Found {len(files)} images.")


# =====================================================
# PROCESS
# =====================================================

for fname in files:

    print(f"\nProcessing {fname}")

    img = cv2.imread(
        os.path.join(INPUT_DIR, fname),
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        print(f"Could not read {fname}")
        continue

    H, W = img.shape

    img = img.astype(np.float32)

    # create a fake scan with one slice
    scan = np.expand_dims(img, axis=0)

    mask = lung_segmentation_scan(
        resUnet,
        scan,
        final_img_size=(W, H)
    )[0]

    # =================================================
    # SAVE MASK
    # =================================================

    mask_uint8 = (mask * 255).astype(np.uint8)

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            fname.replace(".png", "_mask.png")
        ),
        mask_uint8
    )

    # =================================================
    # SAVE OVERLAY
    # =================================================

    overlay = cv2.cvtColor(
        img.astype(np.uint8),
        cv2.COLOR_GRAY2BGR
    )

    overlay[mask > 0] = [0, 0, 255]

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            fname.replace(".png", "_overlay.png")
        ),
        overlay
    )

print("\nDone.")