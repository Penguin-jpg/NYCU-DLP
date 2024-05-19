import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.args = args
        self.device = self.args.device
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(self.device)
        self.optim, self.scheduler = self.configure_optimizers()
        self.state_dict = (
            torch.load(args.ckeckpoint_path) if args.start_from_epoch != 0 else None
        )
        self.prepare_training()

        # set label smoothing to 0.1 like the paper
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.train_losses = []
        self.val_losses = []

    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, train_dataloader):
        self.model.train()

        train_loss = 0

        for i, images in enumerate(tqdm(train_dataloader)):
            images = images.to(self.device)
            logits, z_indices = self.model(images)

            # shape of logits is [batch_size, num_image_tokens, num_codebook_vectors + 1] and
            # the shape of z_indices is [batch_size, num_image_tokens]
            # we need to reshape logits to [batch_size * num_image_tokens, num_codebook_vectors + 1] and
            # reshape z_indices to [batch_size * num_image_tokens] to compute cross-entropy
            logits = logits.view(-1, logits.shape[-1])
            z_indices = z_indices.view(-1)

            loss = self.loss_fn(logits, z_indices)
            # if use gradient accumulation, normalize the loss by gradient accumulation steps
            if self.args.accum_grad != 0:
                loss /= self.args.accum_grad
                loss.backward()

                if (i + 1) % self.args.accum_grad == 0:
                    self.optim.step()
                    # because we use gradient accumulation, we need to clear after the optimizer step
                    self.optim.zero_grad()
            else:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        self.train_losses.append(train_loss)

        return train_loss

    @torch.no_grad()
    def eval_one_epoch(self, val_dataloader):
        self.model.eval()
        os.makedirs("val_results", exist_ok=True)

        mean = torch.tensor([0.4868, 0.4341, 0.3844], device=self.device).view(3, 1, 1)
        std = torch.tensor([0.2620, 0.2527, 0.2543], device=self.device).view(3, 1, 1)
        val_loss = 0
        for i, images in enumerate(tqdm(val_dataloader)):
            images = images.to(self.device)
            logits, z_indices = self.model(images)
            logits = logits.view(-1, logits.shape[-1])
            z_indices = z_indices.view(-1)
            loss = self.loss_fn(logits, z_indices)
            val_loss += loss.item()

            shape = (images.shape[0], 16, 16, 256)
            z_q = self.model.vqgan.codebook.embedding(logits.argmax(dim=-1)).view(shape)
            z_q = z_q.permute(0, 3, 1, 2)
            decoded_image = self.model.vqgan.decode(z_q)
            vutils.save_image(
                (decoded_image * std) + mean,
                os.path.join("val_results", f"image_{i:03d}.png"),
                nrow=images.shape[0],
            )

        val_loss /= len(val_dataloader)
        self.val_losses.append(val_loss)

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.96)
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        return optimizer, scheduler

    def plot_loss(self):
        plt.title("Training and Validation Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.legend()
        plt.savefig("loss.png")
        plt.show()
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MaskGIT")
    # TODO2:check your dataset path is correct
    parser.add_argument(
        "--train_d_path",
        type=str,
        default="./cat_face/train/",
        help="Training Dataset Path",
    )
    parser.add_argument(
        "--val_d_path",
        type=str,
        default="./cat_face/val/",
        help="Validation Dataset Path",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="./checkpoints/last_ckpt.pt",
        help="Path to checkpoint.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Which device the training is on."
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker")
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Batch size for training."
    )
    parser.add_argument(
        "--partial",
        type=float,
        default=1.0,
        help="Number of epochs to train (default: 50)",
    )
    parser.add_argument(
        "--accum-grad", type=int, default=10, help="Number for gradient accumulation."
    )

    # you can modify the hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=0, help="Number of epochs to train."
    )
    parser.add_argument(
        "--save-per-epoch",
        type=int,
        default=1,
        help="Save CKPT per ** epochs(defcault: 1)",
    )
    parser.add_argument(
        "--start-from-epoch", type=int, default=0, help="Number of epochs to train."
    )
    parser.add_argument(
        "--ckpt-interval", type=int, default=0, help="Number of epochs to train."
    )
    parser.add_argument("--learning-rate", type=float, default=0, help="Learning rate.")

    parser.add_argument(
        "--MaskGitConfig",
        type=str,
        default="config/MaskGit.yml",
        help="Configurations for TransformerVQGAN",
    )

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, "r"))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root=args.train_d_path, partial=args.partial)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        shuffle=True,
    )

    val_dataset = LoadTrainData(root=args.val_d_path, partial=args.partial)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        shuffle=False,
    )

    # TODO2 step1-5:
    for epoch in range(args.start_from_epoch + 1, args.epochs + 1):
        train_loss = train_transformer.train_one_epoch(train_loader)
        val_loss = train_transformer.eval_one_epoch(val_loader)
        # step scheduler for each epoch
        train_transformer.scheduler.step()

        if epoch % args.save_per_epoch == 0:
            torch.save(
                train_transformer.model.transformer.state_dict(),
                os.path.join("transformer_checkpoints", f"epoch_{epoch}.pt"),
            )

        print(f"Epoch: {epoch}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")

    train_transformer.plot_loss()
