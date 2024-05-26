import os
import torchvision.transforms as T

from gan import Generator, Discriminator
from losses import generator_loss, discriminator_loss
from dataset import IclevrDataset, TestDataest
from evaluator import evaluation_model
from utils import plot_gan_loss
from sample_gan import inference

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    generator,
    discriminator,
    dataloader,
    generator_optimizer,
    discriminator_optimizer,
    device,
):
    generator.train()
    discriminator.train()

    d_total_loss, g_total_loss = 0, 0
    for image, labels in tqdm(dataloader):
        real = image.to(device)
        labels = labels.to(device)

        # update discriminator
        z = torch.randn([real.shape[0], z_dim]).to(device)
        fake = generator(z, labels).detach()
        # predict on real and fake images
        real_out, real_aux_out = discriminator(real, labels)
        fake_out, fake_aux_out = discriminator(fake, labels)
        d_loss = discriminator_loss(real_out, fake_out, real_aux_out, labels)
        discriminator_optimizer.zero_grad()
        d_loss.backward()
        discriminator_optimizer.step()
        d_total_loss += d_loss.item()

        # update generator
        z = torch.randn([real.shape[0], z_dim]).to(device)
        fake = generator(z, labels)
        fake_out, fake_aux_out = discriminator(fake, labels)
        g_loss = generator_loss(fake_out, fake_aux_out, labels, 1.0)
        generator_optimizer.zero_grad()
        g_loss.backward()
        generator_optimizer.step()
        g_total_loss += g_loss.item()

    d_total_loss /= len(dataloader)
    g_total_loss /= len(dataloader)

    return d_total_loss, g_total_loss


def train(
    generator,
    discriminator,
    eval_model,
    train_dataloader,
    test_dataloader,
    generator_optimizer,
    discriminator_optimizer,
    num_epochs,
    save_interval,
    checkpoint_path,
    results_path,
    device,
):
    d_losses, g_losses = [], []
    best_accuracy = None
    for epoch in range(1, num_epochs + 1):
        d_loss, g_loss = train_one_epoch(
            generator,
            discriminator,
            train_dataloader,
            generator_optimizer,
            discriminator_optimizer,
            device,
        )
        d_losses.append(d_loss)
        g_losses.append(g_loss)

        accuracy = inference(
            generator,
            eval_model,
            test_dataloader,
            os.path.join(results_path, f"{epoch}.png"),
            device,
        )

        print(
            f"Epoch {epoch}/{num_epochs} d_loss: {d_loss} g_loss: {g_loss} accuracy: {accuracy*100:.2f}%"
        )

        if best_accuracy is None or accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(
                generator.state_dict(),
                os.path.join(checkpoint_path, f"generator_best_epoch_{epoch}.pth"),
            )
            print(f"Best accuracy so far: {best_accuracy*100:.2f}%")

        if epoch % save_interval == 0 or epoch == num_epochs - 1:
            torch.save(
                generator.state_dict(),
                os.path.join(checkpoint_path, f"generator_{epoch}.pth"),
            )

    return d_losses, g_losses


if __name__ == "__main__":
    # define some constants
    checkpoint_path = os.path.join("checkpoints", "gan")
    results_path = os.path.join("results", "gan")
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    num_epochs = 1000
    batch_size = 128
    z_dim = 128
    base_channel = 512
    num_classes = 24
    geneartor_lr = 1e-4
    discriminator_lr = 4e-4
    save_interval = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = T.Compose(
        [
            T.Resize((64, 64)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = IclevrDataset(
        dataset_path="iclevr",
        data_json_path="train.json",
        label_json_path="objects.json",
        transform=transform,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_dataest = TestDataest(
        test_json_path="test.json", label_json_path="objects.json"
    )
    test_dataloader = DataLoader(
        test_dataest, batch_size=len(test_dataest), shuffle=False, num_workers=4
    )

    generator = Generator(z_dim, base_channel, num_classes).to(device)
    discriminator = Discriminator(base_channel, num_classes).to(device)
    eval_model = evaluation_model()
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=geneartor_lr, betas=(0, 0.9)
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=discriminator_lr, betas=(0, 0.9)
    )

    d_losses, g_losses = train(
        generator,
        discriminator,
        eval_model,
        train_dataloader,
        test_dataloader,
        generator_optimizer,
        discriminator_optimizer,
        num_epochs,
        save_interval,
        checkpoint_path,
        results_path,
        device,
    )

    plot_gan_loss(d_losses, g_losses)
