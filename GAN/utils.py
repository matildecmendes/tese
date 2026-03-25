import os
import torchvision.utils as vutils

def save_images(fake, epoch, path):
    os.makedirs(path, exist_ok=True)
    vutils.save_image(
        fake,
        f"{path}/epoch_{epoch}.png",
        normalize=True,
        nrow=4
    )