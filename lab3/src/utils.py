import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import pad


def dice_score(pred_mask, gt_mask):
    # dice score = 2 * intersection / (|pred_mask| + |gt_mask|)
    # to calculate intersection of two binary masks, we can compute logical AND of two masks
    # and sum the elements to get the number of pixels in the intersection
    intersection = torch.logical_and(pred_mask, gt_mask).sum(dim=-1)

    # we can also use sum to calculate the number of pixels of the masks
    # return 2 * intersection / (pred_mask.sum(dim=-1) + gt_mask.sum(dim=-1))

    # calculate mask size
    pred_mask_size = pred_mask.shape[-1] * pred_mask.shape[-2]
    gt_mask_size = gt_mask.shape[-1] * gt_mask.shape[-2]

    return 2 * intersection / (pred_mask_size + gt_mask_size)


# image padding
def pad_image(image):
    # reference: https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/5
    width, height = image.size

    # get max side
    max_side = max(width, height)

    # to maintain the aspect ratio of image, we need to pad the shorter side
    width_padding = int((max_side - width) / 2)
    height_padding = int((max_side - height) / 2)

    # left, top, right, bottom
    padding = [width_padding, height_padding, width_padding, height_padding]

    return pad(image, padding, 0, "constant")


def plot_loss(losses, title):
    plt.title(title)
    plt.plot(losses)
    plt.show()
