import os
import argparse
import tensorflow.compat.v1 as tf

from diffusion_tf.tpu_utils.tpu_utils import EvalWorker
from diffusion_tf.tpu_utils import datasets

from scripts.run_lungct import Model
from diffusion_tf.diffusion_utils_2 import get_beta_schedule


def main(args):

    print("\n========== SAMPLING ==========")
    print("checkpoint:", args.ckpt)
    print("num_samples:", args.num_samples)
    print("output_dir:", args.output_dir)
    print("================================\n")

    # =========================
    # Dataset
    # =========================

    dataset = datasets.get_dataset(
        'lungct',
        tfds_data_dir=args.data_dir
    )

    # =========================
    # Diffusion schedule
    # =========================

    betas = get_beta_schedule(
        beta_schedule='linear',
        beta_start=0.0001,
        beta_end=0.02,
        num_diffusion_timesteps=1000
    )

    # =========================
    # Model constructor
    # =========================

    model_constructor = lambda: Model(
        model_name='unet2d16b2',
        betas=betas,
        model_mean_type='eps',
        model_var_type='fixedlarge',
        loss_type='mse',
        num_classes=1,
        dropout=0.0,
        randflip=0
    )

    # =========================
    # Eval worker
    # =========================

    worker = EvalWorker(
        tpu_name=None,
        model_constructor=model_constructor,
        total_bs=args.batch_size,
        dataset=dataset,
        inception_bs=args.batch_size,
        num_inception_samples=args.num_samples
    )

    # =========================
    # Sampling
    # =========================

    worker.run(
        logdir=args.model_dir,
        once=True,
        dump_samples_only=True,
        load_ckpt=args.ckpt,
        samples_dir=args.output_dir
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)

    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument('--data_dir', type=str, required=True)

    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()

    main(args)