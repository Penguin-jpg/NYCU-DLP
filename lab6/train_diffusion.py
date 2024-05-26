from sampler import Diffusion

from diffusion import UNet
from sample_diffusion import inference
from evaluator import evaluation_model
import torch
from torch import optim
from torch.cuda import amp
from tqdm import tqdm
import gc
import os
from torch.utils.data import DataLoader
import torchvision.transforms as T
from dataset import IclevrDataset, TestDataest
from losses import mse_loss


def train_one_epoch(model, diffusion, dataloader, optimizer, scaler, device):
    model.train()

    total_loss = 0
    for image, labels in tqdm(dataloader):
        x_0 = image.to(device)
        labels = labels.to(device)
        timestep_tensor = torch.randint(
            low=1,
            high=diffusion.diffusion_steps,
            size=(x_0.shape[0],),
            device=device,
        )
        x_t, gt_noise = diffusion.forward_process(x_0, timestep_tensor)

        with amp.autocast():
            predicted_noise = model(x_t, timestep_tensor, labels)
            loss = mse_loss(predicted_noise, gt_noise)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    total_loss /= len(dataloader)
    return total_loss


def train(
    model,
    diffusion,
    eval_model,
    train_dataloader,
    test_dataloader,
    optimizer,
    scaler,
    num_epochs,
    results_path,
    checkpoint_path,
    device,
):
    best_accuracy = None
    losses = []
    for epoch in range(1, num_epochs + 1):
        torch.cuda.empty_cache()
        gc.collect()

        loss = train_one_epoch(
            model,
            diffusion,
            train_dataloader,
            optimizer,
            scaler,
            device,
        )
        losses.append(loss)

        accuracy = inference(
            model,
            diffusion,
            eval_model,
            test_dataloader,
            os.path.join(results_path, f"{epoch}.png"),
            device,
        )

        print(f"Epoch {epoch}/{num_epochs} loss: {loss} accuracy: {accuracy*100:.2f}%")

        # only save the best to save disk space
        if best_accuracy is None or accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_path, "model_best.pth"),
            )
            print(f"Best accuracy so far: {best_accuracy*100:.2f}%")

    return losses


if __name__ == "__main__":
    checkpoint_path = os.path.join("checkpoints", "diffusion")
    results_path = os.path.join("results", "diffusion")
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    num_epochs = 500
    batch_size = 128
    lr = 0.00002
    diffusion_steps = 1000
    sampling_steps = 1000
    image_shape = [3, 64, 64]
    eta = 0.0
    num_samples = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = T.Compose(
        [
            T.Resize((64, 64)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )
    train_dataset = IclevrDataset(
        dataset_path="iclevr",
        data_json_path="train.json",
        label_json_path="objects.json",
        use_multi_hot=False,
        transform=transform,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_dataest = TestDataest(
        test_json_path="test.json",
        label_json_path="objects.json",
        use_multi_hot=False,
    )
    test_dataloader = DataLoader(
        test_dataest, batch_size=len(test_dataest), shuffle=False, num_workers=4
    )

    diffusion = Diffusion(
        diffusion_steps,
        sampling_steps,
        image_shape,
        False,
        eta,
        device,
    )

    model = UNet(
        input_channels=3,
        output_channels=3,
        time_input_dim=128,
        num_res_blocks=2,
        base_channels=128,
        base_channels_multiples=(1, 1, 2, 2, 4),
        attention_resoultions=(16,),
        dropout_rate=0.0,
    ).to(device)
    eval_model = evaluation_model()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = amp.GradScaler()

    train(
        model,
        diffusion,
        eval_model,
        train_dataloader,
        test_dataloader,
        optimizer,
        scaler,
        num_epochs,
        results_path,
        checkpoint_path,
        device,
    )
