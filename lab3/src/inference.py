import argparse

import torch
import torch.nn.functional as F
from oxford_pet import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from utils import dice_score, load_model, mask_to_image, save_comparison
import os


def inference(model, dataloader, device):
    model.eval()

    score = 0
    predictions = []
    for image, mask in dataloader:
        image = image.to(device)
        mask = mask.to(device, dtype=torch.long).squeeze(1)

        predicted_mask = model(image)
        for i in range(predicted_mask.shape[0]):
            predictions.append(
                {
                    "image": image[i].detach().cpu().squeeze(0).clip(0, 1).permute(1, 2, 0).numpy(),
                    "mask": mask[i].detach().cpu().squeeze(0).clip(0, 1).numpy(),
                    "pred": mask_to_image(predicted_mask[i].detach().argmax(0).cpu().squeeze(0).clip(0, 1).numpy()),
                }
            )

        score += dice_score(
            F.softmax(predicted_mask, 1).float(),  # turn predicted pixels into probabilities
            F.one_hot(mask, 2)
            .permute(0, 3, 1, 2)
            .float(),  # turn label mask to one-hot encoding to match shape (B, 2, H, W)
        ).item()

    # average over batches
    score /= len(dataloader)

    return score, predictions


def get_args():
    parser = argparse.ArgumentParser(description="Predict masks from input images")
    parser.add_argument("--model", default="MODEL.pth", help="path to the stored model weoght")
    parser.add_argument("--data_path", type=str, help="path to the input data")
    parser.add_argument("--batch_size", "-b", type=int, default=1, help="batch size")
    parser.add_argument("--save_predictions", action="store_true", help="save predicted mask")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_model(args.model, device)
    test_dataset = load_dataset(data_path=args.data_path, mode="test", pad=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    score, predictions = inference(model, test_loader, device)
    print(f"Test dice score: {score:.4f}")

    if args.save_predictions:
        os.makedirs("predictions", exist_ok=True)
        for i, prediction in enumerate(predictions):
            save_comparison(prediction, os.path.join("predictions", f"{i}.png"))
