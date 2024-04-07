import argparse

import torch
import torch.nn.functional as F
from oxford_pet import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from utils import dice_score, load_model, mask_to_image, plot_comparison


def test(model, dataloader, device):
    model.eval()

    score = 0
    predictions = []
    for image, mask in dataloader:
        image = image.to(device)
        mask = mask.to(device, dtype=torch.long).squeeze(1)

        predicted_mask = model(image)
        predictions.append(
            {
                "image": image.detach().cpu().squeeze(0).permute(1, 2, 0).numpy(),
                "mask": mask.detach().cpu().squeeze(0).numpy(),
                "pred": mask_to_image(predicted_mask.detach().argmax(1).cpu().squeeze(0).numpy()),
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

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    test_dataset = load_dataset(data_path=args.data_path, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    loss_fn = nn.CrossEntropyLoss()
    score, predictions = test(model, test_loader, device)
    plot_comparison(predictions[-1])

    print(f"Test dice score: {score:.4f}")
