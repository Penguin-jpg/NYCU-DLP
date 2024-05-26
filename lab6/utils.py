import numpy as np
import random
import torch
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


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


def show_grid_image(images, num_cols, filename="result.png"):
    grid = make_grid(images, nrow=num_cols).cpu()
    save_image(grid, filename)


def show_denoising_process(
    images,
    show_steps=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999],
    filename="denoising.png",
):
    grid = make_grid(images[show_steps], nrow=len(show_steps)).cpu()
    save_image(grid, filename)


def indices_to_multi_hot(indices, num_classes):
    multi_hot = torch.zeros(indices.shape[0], num_classes)
    for i in range(indices.shape[0]):
        for index in indices[i]:
            if index < num_classes:
                multi_hot[i, index] = 1
    return multi_hot
