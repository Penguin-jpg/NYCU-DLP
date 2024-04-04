import argparse
import os

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate import evaluate
from models.unet import UNet
from oxford_pet import load_dataset
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
        for image, mask, _ in train_loader:
            image = image.to(device)
            # remove channel dimension and cast to long (for cross entropy)
            mask = mask.to(device, dtype=torch.long).squeeze(1)
            predicted_mask = model(image)

            loss = loss_fn(predicted_mask, mask) + dice_score(
                F.softmax(
                    predicted_mask, 1
                ).float(),  # turn predicted pixels into probabilities
                F.one_hot(mask, 3)
                .permute(0, 3, 1, 2)
                .float(),  # turn label mask to one-hot encoding to match shape (B, 3, H, W)
            )
            loss = loss_fn(predicted_mask, mask)

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss, _ = evaluate(model, valid_loader, loss_fn, device)

        train_loss /= len(train_loader)
        val_loss /= len(valid_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)

        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )
    parser.add_argument("--data_path", type=str, help="path of the input data")
    parser.add_argument("--epochs", "-e", type=int, default=5, help="number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=1, help="batch size")
    parser.add_argument(
        "--learning-rate", "-lr", type=float, default=1e-5, help="learning rate"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    train_dataset = load_dataset(data_path=args.data_path, mode="train")
    valid_dataset = load_dataset(data_path=args.data_path, mode="valid")
    test_dataset = load_dataset(data_path=args.data_path, mode="test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_multipliers=[1, 2, 4, 8],
    )
    unet.to(device)
    optimizer = optim.SGD(unet.parameters(), lr=args.learning_rate, momentum=0.99)
    loss_fn = nn.CrossEntropyLoss()

    train(
        unet,
        train_loader,
        valid_loader,
        args.epochs,
        loss_fn,
        optimizer,
        os.path.join("saved_models", "UNet.pth"),
        device,
    )
