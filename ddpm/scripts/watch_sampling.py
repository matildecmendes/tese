import numpy as np

from PIL import Image

from scripts.run_lungct import Model
from diffusion_tf.diffusion_utils_2 import get_beta_schedule

import os
import time
import glob
import argparse

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

def save_image(img, path):

    img = np.squeeze(img)

    img = ((img + 1.0) * 127.5)

    img = np.clip(img, 0, 255).astype(np.uint8)

    Image.fromarray(img).save(path)

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

            print("[BUILDING MODEL]")

            betas = get_beta_schedule(
                beta_schedule='linear',
                beta_start=0.0001,
                beta_end=0.02,
                num_diffusion_timesteps=1000
            )

            tf.reset_default_graph()

            model = Model(
                model_name='unet2d16b2',
                betas=betas,
                model_mean_type='eps',
                model_var_type='fixedlarge',
                loss_type='mse',
                num_classes=1,
                dropout=0.0,
                randflip=0
            )

            dummy_x = tf.zeros(
                [1, 256, 256, 1],
                dtype=tf.float32
            )

            dummy_y = tf.zeros(
                [1],
                dtype=tf.int32
            )

            samples = model.samples_fn(
                dummy_x,
                dummy_y
            )['samples']

            saver = tf.train.Saver()

            print("[STARTING SESSION]")

            start = time.time()

            with tf.Session() as sess:

                print("[RESTORING CHECKPOINT]")
                saver.restore(sess, full_ckpt)

                print("[GENERATING SAMPLE]")
                imgs = sess.run(samples)

                out_path = os.path.join(
                    outdir,
                    "sample.png"
                )

                save_image(imgs[0], out_path)

            elapsed = time.time() - start

            print(f"\n[SUCCESS] Saved sample:")
            print(out_path)
            print(f"Time: {elapsed:.2f} sec\n")

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