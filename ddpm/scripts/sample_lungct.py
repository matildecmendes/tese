import os
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image

from scripts.run_lungct import Model
from diffusion_tf.diffusion_utils_2 import get_beta_schedule


def save_image(img, path):

    img = np.squeeze(img)

    img = ((img + 1.0) * 127.5)

    img = np.clip(img, 0, 255).astype(np.uint8)

    Image.fromarray(img).save(path)


def main(args):

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n========== SAMPLING ==========")
    print("checkpoint_dir:", args.checkpoint_dir)
    print("num_samples:", args.num_samples)
    print("output_dir:", args.output_dir)
    print("================================\n")

    latest_ckpt = tf.train.latest_checkpoint(
        args.checkpoint_dir
    )

    if latest_ckpt is None:

        raise ValueError(
            f"No checkpoint found in "
            f"{args.checkpoint_dir}"
        )

    print("\nLatest checkpoint found:")
    print(latest_ckpt)
    print()

    # =========================
    # diffusion schedule
    # =========================

    betas = get_beta_schedule(
        beta_schedule='linear',
        beta_start=0.0001,
        beta_end=0.02,
        num_diffusion_timesteps=1000
    )

    # =========================
    # build model
    # =========================

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

    # dummy inputs
    dummy_x = tf.zeros(
        [args.batch_size, 256, 256, 1],
        dtype=tf.float32
    )

    dummy_y = tf.zeros(
        [args.batch_size],
        dtype=tf.int32
    )

    # sampling graph
    samples = model.samples_fn(
        dummy_x,
        dummy_y
    )['samples']

    # =========================
    # session
    # =========================

    saver = tf.train.Saver()

    with tf.Session() as sess:

        print("\nRestoring checkpoint...\n")

        saver.restore(sess, latest_ckpt)

        print("Checkpoint restored!\n")

        start_idx = 2368 #AQUI MUDAR PARA 0
        total = start_idx

        generated = 0

        while generated < args.num_samples: 

            imgs = sess.run(samples)

            for i in range(imgs.shape[0]):

                if generated >= args.num_samples:
                    break

                out_path = os.path.join(
                    args.output_dir,
                    f"{total:06d}.png"
                )

                save_image(imgs[i], out_path)

                generated += 1
                total += 1

            print(f"Generated {generated}/{args.num_samples}")

    print("\nSampling finished!\n")


if __name__ == '__main__':


    print("Started sampling...")

    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_dir', type=str, required=True)

    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument('--num_samples', type=int, default=100)

    parser.add_argument('--batch_size', type=int, default=4)

    args = parser.parse_args()

    main(args)