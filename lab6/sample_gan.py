from dataset import TestDataest
from gan import Generator
from evaluator import evaluation_model
from utils import seed_everything, denormalize_to_0_and_1, show_grid_image
import os

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def inference(generator, eval_model, dataloader, filename, device):
    for labels in dataloader:
        z = torch.randn([labels.shape[0], generator.z_dim]).to(device)
        labels = labels.to(device)
        generated = generator(z, labels)
        accuracy = eval_model.eval(generated, labels)
        show_grid_image(
            denormalize_to_0_and_1(generated, mean=0.5, std=0.5),
            num_rows=4,
            filename=filename,
        )

    return accuracy


if __name__ == "__main__":
    seed_everything(42)

    z_dim = 128
    base_channel = 512
    num_classes = 24
    results_path = os.path.join("test_results", "gan")
    os.makedirs(results_path, exist_ok=True)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(z_dim, base_channel, num_classes)
    generator.load_state_dict(
        torch.load(
            os.path.join("checkpoints", "gan", "generator_best.pth"), map_location="cpu"
        )
    )
    generator.to(device)
    generator.eval()

    eval_model = evaluation_model()

    accuracy1 = inference(
        generator,
        eval_model,
        test_dataloader1,
        os.path.join(results_path, "test.png"),
        device,
    )
    accuracy2 = inference(
        generator,
        eval_model,
        test_dataloader1,
        os.path.join(results_path, "new_test.png"),
        device,
    )
    print(f"Accuracy of test.json: {accuracy1*100:.2f}%")
    print(f"Accuracy of new_test.json: {accuracy2*100:.2f}%")
