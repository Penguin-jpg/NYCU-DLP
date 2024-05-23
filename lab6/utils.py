import numpy as np
import random
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def plot_gan_loss(g_losses, d_losses):
    plt.title("GAN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.legend()
    plt.savefig("gan_loss.png")
    plt.show()
    plt.clf()


def denormalize_to_0_and_1(image, mean=None, std=None):
    # for diffusion
    if mean is None and std is None:
        return (image + 1.0) / 2.0
    # for gan
    return image * std + mean


def show_grid_image(images, num_rows, filename="result.png"):
    grid = make_grid(images, nrow=num_rows).cpu().permute(2, 1, 0).numpy()
    plt.imshow(grid)
    plt.savefig(filename)
    plt.show()
    plt.clf()
