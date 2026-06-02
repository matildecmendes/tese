import json
import subprocess
import sys

TRIAL_ID = int(sys.argv[1])

BASE_OUTPUT_DIR = "/nas-ctm01/homes/mcmendes/outputs/random_search"

REAL_IMG_DIR = "/nas-ctm01/datasets/public/lidc-mcmendes/slices_png"

DATA_PATH = "/nas-ctm01/datasets/public/lidc-mcmendes/slices"

with open("trial_configs.json", "r") as f:
    trials = json.load(f)

cfg = trials[str(TRIAL_ID)]

print(f"Running trial {TRIAL_ID}")
print(cfg)

cmd = [

    "python",

    "ddgan_randoms.py",

    "--trial_id", str(TRIAL_ID),

    "--trial_config", "trial_configs.json",

    "--data_path", DATA_PATH,

    "--real_img_dir", REAL_IMG_DIR,

    "--output_dir", BASE_OUTPUT_DIR,

    "--batch_size", "4",

    "--num_epoch", "25",

    "--num_channels", "1",

    "--image_size", "256",

    "--ch_mult", "1", "2", "2", "4",

    "--dataset", "custom",

    "--save_ckpt_every", "5",

    "--num_fid_samples", "500"
]

subprocess.run(cmd)