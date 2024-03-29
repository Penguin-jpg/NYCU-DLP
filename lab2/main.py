import torch
from torch.utils.data import DataLoader

from dataloader import BufferflyMothLoader
from VGG19 import VGG19
from ResNet50 import ResNet50

import matplotlib.pyplot as plt

import os


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
    train_accuracy = []
    test_accuracy = []
    best_loss = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        accuracy = 0
        for data, label in train_dataloader:
            data = data.to(device)
            label = label.to(device)
            predicted = model(data)
            accuracy += (predicted.argmax(1) == label).type(torch.float).sum().item()
            loss = loss_fn(predicted, label)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        accuracy /= len(train_dataloader.dataset)
        total_loss /= len(train_dataloader)
        train_accuracy.append(accuracy)
        losses.append(total_loss)
        test_accuracy.append(evaluate(model, test_dataloader, device))

        if best_loss is None or total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), os.path.join(model_path, "best.pt"))
            print(f"Model saved at {model_path}")

        print(f"Epoch {epoch} loss: {total_loss}")

    return losses, train_accuracy, test_accuracy


def load_model(model_name, model_path, device):
    if model_name == "vgg19":
        model = VGG19()
    else:
        model = ResNet50()

    model.load_state_dict(torch.load(os.path.join(model_path, "best.pt"), map_location="cpu"))
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


def plot_accuracy(vgg_train_accuracy, vgg_test_accuracy, resnet_train_accuracy, resnet_test_accuracy):
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
    BATCH_SIZE = 128
    NUM_EPOCHS = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = BufferflyMothLoader(root="dataset", mode="train")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_dataset = BufferflyMothLoader(root="dataset", mode="test")
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # train vgg19
    vgg = VGG19()
    vgg.to(device)
    vgg_losses, vgg_train_accuracy, vgg_test_accuracy = train(
        vgg,
        train_dataloader,
        test_dataloader,
        NUM_EPOCHS,
        torch.nn.CrossEntropyLoss(),
        # torch.optim.Adam(vgg.parameters(), lr=1e-2),
        torch.optim.SGD(vgg.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4),
        os.path.join("models", "vgg19"),
        device,
    )
    # print_result("vgg19", max(vgg_train_accuracy), max(vgg_test_accuracy))

    # # train resnet50
    # resnet = ResNet50()
    # resnet.to(device)
    # resnet_losses, resnet_train_accuracy, resnet_test_accuracy = train(
    #     resnet,
    #     train_dataloader,
    #     test_dataloader,
    #     NUM_EPOCHS,
    #     torch.nn.CrossEntropyLoss(),
    #     torch.optim.Adam(resnet.parameters(), lr=1e-4),
    #     os.path.join("models", "resnet50"),
    #     device,
    # )

    # print_result("resnet50", max(resnet_train_accuracy), max(resnet_test_accuracy))

    # # plot accuracy
    # plot_accuracy(
    #     vgg_train_accuracy,
    #     vgg_test_accuracy,
    #     resnet_train_accuracy,
    #     resnet_test_accuracy,
    # )
