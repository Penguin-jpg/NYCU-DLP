import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluate import evaluate
from oxford_pet import load_dataset
from utils import load_model, plot_comparison


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
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    test_loss, predictions = evaluate(model, test_loader, loss_fn, device, testing=True)
    plot_comparison(predictions[-1])

    print(f"Test loss: {test_loss:.4f}")
