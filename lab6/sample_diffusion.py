from dataset import TestDataest
from diffusion import UNet
from sampler import Diffusion
from evaluator import evaluation_model
from utils import (
    seed_everything,
    indices_to_multi_hot,
    denormalize_to_0_and_1,
    show_grid_image,
)
import os

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T


@torch.no_grad()
def inference(model, diffusion, eval_model, dataloader, filename, device):
    normalize = T.Compose([T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    for labels in dataloader:
        labels = labels.to(device)
        generated = diffusion.sample(model, labels, num_samples=labels.shape[0])
        generated = denormalize_to_0_and_1(generated)
        accuracy = eval_model.eval(
            normalize(generated), indices_to_multi_hot(labels, 24)
        )
        show_grid_image(
            generated,
            num_rows=4,
            filename=filename,
        )

    return accuracy


if __name__ == "__main__":
    seed_everything(15)

    results_path = os.path.join("test_results", "diffusion")
    os.makedirs(results_path, exist_ok=True)

    num_epochs = 100
    batch_size = 128
    lr = 0.00002
    save_interval = 2
    diffusion_steps = 1000
    sampling_steps = 1000
    image_shape = [3, 64, 64]
    eta = 0.0
    num_samples = 16
    schedule = "linear"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset1 = TestDataest(
        test_json_path="test.json",
        label_json_path="objects.json",
        use_multi_hot=False,
    )
    test_dataloader1 = DataLoader(
        test_dataset1, batch_size=len(test_dataset1), shuffle=False
    )
    test_dataset2 = TestDataest(
        test_json_path="new_test.json",
        label_json_path="objects.json",
        use_multi_hot=False,
    )
    test_dataloader1 = DataLoader(
        test_dataset2, batch_size=len(test_dataset2), shuffle=False
    )

    diffusion = Diffusion(
        diffusion_steps,
        sampling_steps,
        image_shape,
        False,
        eta,
        schedule,
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
    )
    model.load_state_dict(
        torch.load(
            os.path.join("checkpoints", "diffusion", "model_best.pth"),
            map_location="cpu",
        )
    )
    model.to(device)
    model.eval()

    eval_model = evaluation_model()

    accuracy1 = inference(
        model,
        diffusion,
        eval_model,
        test_dataloader1,
        os.path.join(results_path, "test.png"),
        device,
    )
    accuracy2 = inference(
        model,
        diffusion,
        eval_model,
        test_dataloader1,
        os.path.join(results_path, "new_test.png"),
        device,
    )
    print(f"Accuracy of test.json: {accuracy1*100:.2f}%")
    print(f"Accuracy of new_test.json: {accuracy2*100:.2f}%")
