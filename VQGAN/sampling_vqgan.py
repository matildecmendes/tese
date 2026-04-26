import argparse
import os
import torch
import torchvision
from omegaconf import OmegaConf

from taming.models.cond_transformer import Net2NetTransformer
from main import instantiate_from_config

# =========================
# Load model
# =========================

def load_model(config_path, ckpt_path, device):
    print("Loading model...")

    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)

    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(sd, strict=False)

    model = model.to(device)
    model.eval()

    return model


# =========================
# Sampling
# =========================

@torch.no_grad()
def generate_images(model, args, device):
    os.makedirs(args.output_dir, exist_ok=True)

    total = 0
    n_batches = args.num_samples // args.batch_size

    print("Start sampling...")

    for i in range(n_batches):

        # 🔥 sampling de tokens
        z_indices = model.sample(
            batch_size=args.batch_size,
            cond=None,
            steps=256,           # 16x16 tokens
            temperature=1.0,
            sample=True,
            top_k=100
        )

        # 🔥 decode com VQGAN
        quant_z = model.first_stage_model.quantize.get_codebook_entry(
            z_indices.view(-1), shape=(args.batch_size, 16, 16, 256)
        )

        img = model.first_stage_model.decode(quant_z)

        # [-1,1] → [0,1]
        img = (img + 1) / 2
        img = img.clamp(0, 1)

        for j in range(img.size(0)):
            idx = i * args.batch_size + j
            path = os.path.join(args.output_dir, f"{idx}.png")
            torchvision.utils.save_image(img[j], path)

        total += img.size(0)

        if i % 10 == 0:
            print(f"Generated {total}/{args.num_samples}")

    print("Sampling finished!")


# =========================
# Main
# =========================

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(args.config_path, args.ckpt_path, device)

    generate_images(model, args, device)


# =========================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()

    main(args)