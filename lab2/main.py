import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import ButterflyMothLoader
from ResNet50 import ResNet50
from VGG19 import VGG19


def evaluate(model, dataloader, device):
    model.eval()

    accuracy = 0
    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            label = label.to(device)
            predicted = model(data)

            # if correct, accuracy will be increased by 1
            accuracy += (predicted.argmax(1) == label).type(torch.float).sum().item()

    accuracy /= len(dataloader.dataset)
    return accuracy


def test(model, dataloader, device):
    return evaluate(model, dataloader, device)


def train(
    model,
    train_dataloader,
    test_dataloader,
    num_epochs,
    loss_fn,
    optimizer,
    model_path,
    device,
):
    losses = []
    train_accuracy_list = []
    test_accuracy_list = []
    best_accuracy = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_accuracy = 0
        for data, label in train_dataloader:
            data = data.to(device)
            label = label.to(device)
            predicted = model(data)
            train_accuracy += (
                (predicted.argmax(1) == label).type(torch.float).sum().item()
            )
            loss = loss_fn(predicted, label)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_accuracy /= len(train_dataloader.dataset)
        total_loss /= len(train_dataloader)
        train_accuracy_list.append(train_accuracy)
        losses.append(total_loss)
        test_accuracy = evaluate(model, test_dataloader, device)
        test_accuracy_list.append(test_accuracy)

        # information to store
        state_dict = {
            "epoch": epoch,
            "model": model.state_dict(),
            "train_accuracy": train_accuracy_list,
            "test_accuracy": test_accuracy_list,
        }

        if best_accuracy is None or test_accuracy < best_accuracy:
            best_accuracy = test_accuracy
            torch.save(state_dict, os.path.join(model_path, "best.pt"))
            print(f"Model saved at {model_path}")
        elif epoch == num_epochs - 1:
            torch.save(state_dict, os.path.join(model_path, f"{epoch}.pt"))
            print(f"Model saved at {model_path}")

        print(
            f"Epoch {epoch} loss: {total_loss:.5f} train_accuracy: {train_accuracy*100:.2f} test_accuracy: {test_accuracy*100:.2f}"
        )

    return losses, train_accuracy_list, test_accuracy_list


def load_model(model_name, model_path, device):
    if model_name == "vgg19":
        model = VGG19()
    else:
        model = ResNet50()

    model.load_state_dict(
        torch.load(os.path.join(model_path, "best.pt"), map_location="cpu")
    )
    model.to(device)
    model.eval()
    return model


def print_result(model_name, best_train_accuracy, best_test_accuracy):
    if model_name == "vgg19":
        print("----------VGG19-----------")
        print(
            f"VGG19          |    Train accuracy:   {best_train_accuracy*100:.2f}%|    Test accuracy:   {best_test_accuracy*100:.2f}%"
        )
    else:
        print("----------ResNet50-----------")
        print(
            f"ResNet50          |    Train accuracy:   {best_train_accuracy*100:.2f}%|    Test accuracy:   {best_test_accuracy*100:.2f}%"
        )


def plot_accuracy(
    vgg_train_accuracy, vgg_test_accuracy, resnet_train_accuracy, resnet_test_accuracy
):
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(vgg_train_accuracy, label="VGG19_train_acc")
    plt.plot(vgg_test_accuracy, label="VGG19_test_acc")
    plt.plot(resnet_train_accuracy, label="ResNet50_train_acc")
    plt.plot(resnet_test_accuracy, label="ResNet50_test_acc")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    BATCH_SIZE = 64
    NUM_EPOCHS = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),  # random flip
            transforms.ToTensor(),  # image to tensor and normalize to [0, 1]
            transforms.Lambda(lambda x: x * 2 - 1),  # normalize to [-1, 1]
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = ButterflyMothLoader(
        root="dataset", mode="train", transform=train_transform
    )
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = ButterflyMothLoader(
        root="dataset", mode="test", transform=test_transform
    )
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # train vgg19
    # vgg = VGG19()
    # vgg.to(device)
    # vgg_losses, vgg_train_accuracy, vgg_test_accuracy = train(
    #     vgg,
    #     train_dataloader,
    #     test_dataloader,
    #     NUM_EPOCHS,
    #     torch.nn.CrossEntropyLoss(),
    #     # torch.optim.Adam(vgg.parameters(), lr=1e-2),
    #     torch.optim.SGD(vgg.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4),
    #     os.path.join("models", "vgg19"),
    #     device,
    # )
    # print_result("vgg19", max(vgg_train_accuracy), max(vgg_test_accuracy))

    # train resnet50
    resnet = ResNet50()
    resnet.to(device)
    resnet_losses, resnet_train_accuracy, resnet_test_accuracy = train(
        resnet,
        train_dataloader,
        test_dataloader,
        NUM_EPOCHS,
        torch.nn.CrossEntropyLoss(),
        torch.optim.Adam(resnet.parameters(), lr=1e-4),
        # torch.optim.SGD(resnet.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4),
        os.path.join("models", "resnet50"),
        device,
    )

    print_result("resnet50", max(resnet_train_accuracy), max(resnet_test_accuracy))

    # plot accuracy
    plot_accuracy(
        resnet_train_accuracy,
        resnet_test_accuracy,
        resnet_train_accuracy,
        resnet_test_accuracy,
    )
