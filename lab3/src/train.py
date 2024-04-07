import argparse
import os

import torch
import torch.nn.functional as F
from evaluate import evaluate
from models.resnet34_unet import ResNet34Unet
from models.unet import UNet
from oxford_pet import load_dataset
from torch import nn, optim
from torch.utils.data import DataLoader
from utils import dice_score, plot_loss


def train(
    model,
    train_loader,
    valid_loader,
    num_epochs,
    loss_fn,
    optimizer,
    model_path,
    device,
):
    train_losses = []
    val_losses = []

    best_val_loss = None

    for epoch in range(num_epochs):
        model.train()

        train_loss = 0
        for image, mask in train_loader:
            image = image.to(device)
            # remove channel dimension and cast to long (for cross entropy)
            mask = mask.to(device, dtype=torch.long).squeeze(1)
            predicted_mask = model(image)
            loss = loss_fn(predicted_mask, mask) + (
                1
                - dice_score(
                    F.softmax(predicted_mask, 1).float(),  # turn predicted pixels into probabilities
                    F.one_hot(mask, 2)
                    .permute(0, 3, 1, 2)
                    .float(),  # turn label mask to one-hot encoding to match shape (B, 2, H, W)
                )
            )

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = evaluate(model, valid_loader, loss_fn, device)

        train_loss /= len(train_loader)
        val_loss /= len(valid_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at {model_path}")

        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    plot_loss(train_losses, val_losses)


def get_args():
    parser = argparse.ArgumentParser(description="Train the UNet on images and target masks")
    parser.add_argument("--model", type=str, default="unet", help="choose unet or resnet_unet to train")
    parser.add_argument("--data_path", type=str, help="path of the input data")
    parser.add_argument("--epochs", "-e", type=int, default=5, help="number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=1, help="batch size")
    parser.add_argument("--learning-rate", "-lr", type=float, default=1e-5, help="learning rate")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    train_dataset = load_dataset(data_path=args.data_path, mode="train")
    valid_dataset = load_dataset(data_path=args.data_path, mode="valid")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "unet":
        model = UNet(
            in_channels=3,
            out_channels=2,
            base_channels=64,
            channel_multipliers=[1, 2, 4, 8],
        )
    else:
        model = ResNet34Unet(
            in_channels=3,
            out_channels=2,
        )

    model.to(device)

    # optimizer = optim.SGD(unet.parameters(), lr=args.learning_rate, momentum=0.99)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    train(
        model,
        train_loader,
        valid_loader,
        args.epochs,
        loss_fn,
        optimizer,
        os.path.join("saved_models", "UNet.pth"),
        device,
    )
