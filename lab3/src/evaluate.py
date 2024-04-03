from utils import dice_score


def evaluate(model, dataloader, loss_fn, device):
    model.eval()

    val_loss = 0
    for sample in dataloader:
        image = sample["image"].to(device)
        mask = sample["mask"].to(device)

        predicted_mask = model(image)
        loss = loss_fn(predicted_mask, mask) + dice_score(predicted_mask, mask)

        val_loss += loss.item()

    return val_loss
