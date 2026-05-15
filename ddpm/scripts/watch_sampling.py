import os
import time
import glob
import argparse
import subprocess

import tensorflow.compat.v1 as tf
from tensorflow.python.client import device_lib


def print_device_info():

    print("\n========== DEVICE CHECK ==========")

    print("TF version:", tf.__version__)

    print("Built with CUDA:", tf.test.is_built_with_cuda())

    print("GPU available:", tf.test.is_gpu_available())

    print("\nLocal devices:\n")
    print(device_lib.list_local_devices())

    print("\nCUDA_VISIBLE_DEVICES:",
          os.environ.get("CUDA_VISIBLE_DEVICES"))

    print("==================================\n")

    if not tf.test.is_gpu_available():

        print("\n[WARNING] GPU NOT DETECTED!")
        print("Sampling will run VERY slowly on CPU.")
        print("Check CUDA / cuDNN / TensorFlow versions.\n")


def main(args):

    print_device_info()

    os.makedirs(args.output_dir, exist_ok=True)

    already_done = set()

    print("\n========== WATCHER START ==========")
    print("checkpoint_dir:", args.checkpoint_dir)
    print("output_dir:", args.output_dir)
    print("poll_every:", args.poll_every)
    print("===================================\n")

    while True:

        ckpts = sorted(
            glob.glob(
                os.path.join(
                    args.checkpoint_dir,
                    "model.ckpt-*.index"
                )
            )
        )

        if len(ckpts) == 0:

            print("[WAITING] No checkpoints found yet...")
            time.sleep(args.poll_every)
            continue

        for ckpt in ckpts:

            ckpt_name = os.path.basename(
                ckpt
            ).replace(".index", "")

            step = int(ckpt_name.split("-")[-1])

            if step < args.min_step:
                print(f"[SKIPPING] {ckpt_name} (< min_step)")
                continue

            if ckpt_name in already_done:
                continue

            already_done.add(ckpt_name)

            full_ckpt = os.path.join(
                args.checkpoint_dir,
                ckpt_name
            )

            print("\n===================================")
            print("[NEW CHECKPOINT DETECTED]")
            print(full_ckpt)
            print("===================================\n")

            outdir = os.path.join(
                args.output_dir,
                ckpt_name
            )

            os.makedirs(outdir, exist_ok=True)

            cmd = [
                "python",
                "scripts/sample_lungct.py",

                "--ckpt_path",
                full_ckpt,

                "--output_dir",
                outdir,

                "--num_samples",
                "1",

                "--batch_size",
                "1"
            ]

            print("[RUNNING SAMPLING]")
            print(" ".join(cmd))
            print()

            start = time.time()

            result = subprocess.run(cmd)

            elapsed = time.time() - start

            if result.returncode == 0:

                print(f"\n[SUCCESS] Sampling finished in "
                      f"{elapsed:.2f} sec")

            else:

                print("\n[ERROR] Sampling crashed!")

        print(f"\n[SLEEPING {args.poll_every} sec]\n")

        time.sleep(args.poll_every)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--poll_every",
        type=int,
        default=60
    )

    parser.add_argument(
        "--min_step",
        type=int,
        default=0
    )

    args = parser.parse_args()

    main(args)