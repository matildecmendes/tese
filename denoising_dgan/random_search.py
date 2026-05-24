import os
import csv
import random
import subprocess

# ==========================================
# CONFIGURAÇÕES GERAIS
# ==========================================

NUM_TRIALS = 1

BASE_OUTPUT_DIR = "/nas-ctm01/homes/mcmendes/outputs/random_search"

TRAIN_SCRIPT = "ddgan_randoms.py"

REAL_IMG_DIR = "/nas-ctm01/datasets/public/lidc-mcmendes/slices_png"

DATA_PATH = "/nas-ctm01/datasets/public/lidc-mcmendes/slices"

# ==========================================
# SEARCH SPACE
# ==========================================

LR_G_VALUES = [5e-5, 1e-4, 1.5e-4, 2e-4]

LR_D_VALUES = [5e-5, 1e-4, 1.5e-4]

R1_VALUES = [0.01, 0.05, 0.1]

TIMESTEP_VALUES = [2, 4, 8]

# ==========================================
# CSV RESULTS
# ==========================================

csv_path = os.path.join(BASE_OUTPUT_DIR, "results.csv")

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

if not os.path.exists(csv_path):

    with open(csv_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "trial",
            "lr_g",
            "lr_d",
            "r1_gamma",
            "num_timesteps",
            "fid"
        ])

# ==========================================
# RANDOM SEARCH LOOP
# ==========================================

for trial in range(NUM_TRIALS):

    print(f"\n===== STARTING TRIAL {trial} =====")

    # --------------------------------------
    # Random parameters
    # --------------------------------------

    lr_g = random.choice(LR_G_VALUES)

    lr_d = random.choice(LR_D_VALUES)

    r1_gamma = random.choice(R1_VALUES)

    num_timesteps = random.choice(TIMESTEP_VALUES)

    # --------------------------------------
    # Trial output dir
    # --------------------------------------

    exp_name = f"trial_{trial:03d}"

    output_dir = os.path.join(BASE_OUTPUT_DIR, exp_name)

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------
    # Training command
    # --------------------------------------

    cmd = [

        "python",

        TRAIN_SCRIPT,

        "--data_path", DATA_PATH,

        "--real_img_dir", REAL_IMG_DIR,

        "--output_dir", BASE_OUTPUT_DIR,

        "--exp", exp_name,

        "--batch_size", "8",

        "--num_epoch", "5",

        "--num_channels", "1",

        "--image_size", "256",

        "--ch_mult", "1", "2", "2", "4",

        "--dataset", "custom",

        "--save_ckpt_every", "5",

        "--num_fid_samples", "100",

        "--lr_g", str(lr_g),

        "--lr_d", str(lr_d),

        "--r1_gamma", str(r1_gamma),

        "--num_timesteps", str(num_timesteps)
    ]

    # --------------------------------------
    # Run training
    # --------------------------------------

    subprocess.run(cmd)

    # --------------------------------------
    # Read FID
    # --------------------------------------

    fid_file = os.path.join(output_dir, "fid.txt")

    if os.path.exists(fid_file):

        with open(fid_file, "r") as f:

            fid = float(f.read().strip())

    else:

        fid = -1

    print(f"TRIAL {trial} FID: {fid}")

    # --------------------------------------
    # Save CSV
    # --------------------------------------

    with open(csv_path, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            exp_name,
            lr_g,
            lr_d,
            r1_gamma,
            num_timesteps,
            fid
        ])

print("\n===== RANDOM SEARCH FINISHED =====")