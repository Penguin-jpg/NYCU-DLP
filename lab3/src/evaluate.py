import torch
import torch.nn.functional as F

from utils import dice_score, mask_to_image


def evaluate(model, dataloader, loss_fn, device):
    model.eval()

    val_loss = 0
    for image, mask, _ in dataloader:
        image = image.to(device)
        mask = mask.to(device, dtype=torch.long).squeeze(1)

        predicted_mask = model(image)
        loss = loss_fn(predicted_mask, mask) - dice_score(
            F.softmax(
                predicted_mask, 1
            ).float(),  # turn predicted pixels into probabilities
            F.one_hot(mask, 3)
            .permute(0, 3, 1, 2)
            .float(),  # turn label mask to one-hot encoding to match shape (B, 3, H, W)
        )

        val_loss += loss.item()

    return val_loss


def test(model, dataloader, device):
    model.eval()

    score = 0
    predictions = []
    for image, mask, _ in dataloader:
        image = image.to(device)
        mask = mask.to(device, dtype=torch.long).squeeze(1)

        predicted_mask = model(image)
        predictions.append(
            {
                "image": image.detach().cpu().squeeze(0).permute(1, 2, 0).numpy(),
                "mask": mask.detach().cpu().squeeze(0).numpy(),
                "pred": mask_to_image(
                    predicted_mask.detach().argmax(1).cpu().squeeze(0).numpy()
                ),
            }
        )

        score += dice_score(
            F.softmax(
                predicted_mask, 1
            ).float(),  # turn predicted pixels into probabilities
            F.one_hot(mask, 3)
            .permute(0, 3, 1, 2)
            .float(),  # turn label mask to one-hot encoding to match shape (B, 3, H, W)
        )

    # average over batches
    score /= len(dataloader)

    return score, predictions
