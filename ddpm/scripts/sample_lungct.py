import os
import time

import numpy as np
import tensorflow.compat.v1 as tf

from diffusion_tf import utils
from diffusion_tf.tpu_utils.tpu_utils import unnormalize_data

from scripts.run_lungct import Model

tf.disable_v2_behavior()


CHECKPOINT_DIR = "/CAMINHO/PARA/O/MODEL_DIR"

OUTPUT_DIR = os.path.join(CHECKPOINT_DIR, "samples_debug")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():

    model = Model()

    noise = tf.random_normal([8, 256, 256, 1])

    samples = model.samples_fn(
        noise,
        tf.zeros([8], dtype=tf.int32)
    )['samples']

    saver = tf.train.Saver()

    last_ckpt = None

    with tf.Session() as sess:

        while True:

            ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)

            if ckpt is None:
                print("No checkpoint found...")
                time.sleep(30)
                continue

            if ckpt != last_ckpt:

                print(f"\n[NEW CHECKPOINT] {ckpt}")

                saver.restore(sess, ckpt)

                imgs = sess.run(samples)

                imgs = np.clip(
                    unnormalize_data(imgs),
                    0,
                    255
                ).astype(np.uint8)

                step = ckpt.split("-")[-1]

                out_path = os.path.join(
                    OUTPUT_DIR,
                    f"sample_{step}.png"
                )

                utils.save_tiled_imgs(
                    out_path,
                    imgs,
                    num_col=4
                )

                print(f"[SAVED] {out_path}")

                last_ckpt = ckpt

            time.sleep(30)


if __name__ == "__main__":
    main()