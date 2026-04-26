import argparse
import os
import torch
import torchvision
from omegaconf import OmegaConf

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
# Sampling (CORRETO)
# =========================

@torch.no_grad()
def sample_tokens(model, batch_size, device, steps=256, temperature=1.0, top_k=100):

    z = torch.zeros(batch_size, 1, dtype=torch.long).to(device)

    for step in range(steps):

        logits, _ = model.transformer(z)

        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, ix = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = torch.nn.functional.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        z = torch.cat((z, next_token), dim=1)

    return z


# =========================
# Decode
# =========================

@torch.no_grad()
def decode_tokens(model, z_indices):

    z_indices = z_indices[:, 1:]
    b = z_indices.shape[0]

    z_indices = z_indices.view(b, 16, 16)

    quant_z = model.first_stage_model.quantize.get_codebook_entry(
        z_indices.reshape(-1),
        shape=(b, 16, 16, model.first_stage_model.quantize.e_dim)
    )

    img = model.first_stage_model.decode(quant_z)

    return img


# =========================
# Generate images
# =========================

@torch.no_grad()
def generate_images(model, args, device):

    os.makedirs(args.output_dir, exist_ok=True)

    total = 0
    n_batches = args.num_samples // args.batch_size

    print("Start sampling...")

    for i in range(n_batches):

        z_indices = sample_tokens(
            model,
            args.batch_size,
            device,
            steps=256,
            temperature=args.temperature,
            top_k=args.top_k
        )

        img = decode_tokens(model, z_indices)

        img = (img + 1) / 2
        img = img.clamp(0, 1)

        for j in range(img.size(0)):
            idx = i * args.batch_size + j
            path = os.path.join(args.output_dir, f"{idx}.png")
            torchvision.utils.save_image(img[j], path)

        total += img.size(0)

        if i % 5 == 0:
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
    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=100)

    args = parser.parse_args()

    main(args)