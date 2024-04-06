import matplotlib.pyplot as plt
import numpy as np
import torch
from models.unet import UNet
from models.resnet34_unet import ResNet34Unet
from torchvision.transforms.functional import pad


# reference:
# 1. https://github.com/milesial/Pytorch-UNet/blob/master/utils/dice_score.py
# 2. https://blog.csdn.net/ODIMAYA/article/details/123844795
def binary_dice_score(pred_mask, gt_mask):
    # dice score = 2 * intersection / (|pred_mask| + |gt_mask|) (denominator is union)
    B, H, W = pred_mask.shape

    # since we consider the whole image, we can reduce the height and width into a single dimension
    # to simplify computation
    pred = pred_mask.view(B, -1)
    gt = gt_mask.view(B, -1)

    # to calculate intersection of two binary masks, we can compute simply mutlipy two masks
    # and sum the elements to get the number of pixels in the intersection
    intersection = (pred * gt).sum(1)

    # to calculate union of two binary masks, we can compute the sum of both masks
    union = pred.sum(1) + gt.sum(1)

    # average over batch
    return (2 * intersection / union).mean()


def dice_score(pred_mask, gt_mask):
    B, C, H, W = pred_mask.shape

    score = 0

    # calculate dice score for each channel and average over channel
    for i in range(C):
        score += binary_dice_score(pred_mask[:, i], gt_mask[:, i])
    score /= C

    return score


# reference: https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/5
def pad_image(image):
    width, height = image.size

    # get max side
    max_side = max(width, height)

    # to maintain the aspect ratio of image, we need to pad the shorter side
    width_padding = int((max_side - width) / 2)
    height_padding = int((max_side - height) / 2)

    # left, top, right, bottom
    padding = [width_padding, height_padding, width_padding, height_padding]

    return pad(image, padding, 0, "constant")


def plot_loss(train_losses, val_losses):
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.savefig("loss.png")
    plt.show()


def mask_to_image(mask):
    mask = mask.astype(np.float32)
    mask[mask == 2.0] = 0.0
    mask[(mask == 1.0) | (mask == 3.0)] = 1.0
    return mask


def plot_comparison(predicition):
    plt.suptitle("Test Results")
    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(predicition["image"])
    plt.subplot(1, 3, 2)
    plt.title("GT")
    plt.imshow(predicition["mask"])
    plt.subplot(1, 3, 3)
    plt.title("Prediction")
    plt.imshow(predicition["pred"])
    plt.tight_layout()
    plt.savefig("prediction.png")
    plt.show()


def load_model(model_path, device):
    state_dict = torch.load(model_path, map_location="cpu")
    if "ResNet34" in model_path:
        model = ResNet34Unet(in_channels=3, out_channels=2)
    else:
        model = UNet(
            in_channels=3,
            out_channels=2,
            base_channels=64,
            channel_multipliers=[1, 2, 4, 8],
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
