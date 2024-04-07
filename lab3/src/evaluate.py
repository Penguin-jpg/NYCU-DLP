import torch
import torch.nn.functional as F
from utils import dice_score


def evaluate(model, dataloader, loss_fn, device):
    model.eval()

    val_loss = 0
    for image, mask in dataloader:
        image = image.to(device)
        mask = mask.to(device, dtype=torch.long).squeeze(1)

        predicted_mask = model(image)
        loss = loss_fn(predicted_mask, mask) + (
            1
            - dice_score(
                F.softmax(predicted_mask, 1).float(),  # turn predicted pixels into probabilities
                F.one_hot(mask, 2)
                .permute(0, 3, 1, 2)
                .float(),  # turn label mask to one-hot encoding to match shape (B, 3, H, W)
            )
        )

        val_loss += loss.item()

    return val_loss
