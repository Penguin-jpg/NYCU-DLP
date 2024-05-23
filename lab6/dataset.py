from torch.utils.data import Dataset
import os
from glob import glob
from PIL import Image
import torch
import torchvision.transforms as T
import json


class IclevrDataset(Dataset):
    def __init__(self, dataset_path, data_json_path, label_json_path, transform=None):
        super(IclevrDataset, self).__init__()

        self.data_table = json.load(open(data_json_path, "r"))
        self.label_table = json.load(open(label_json_path, "r"))

        # since not all images in dataset_path have labels in data_table, we only select the images that have labels
        self.image_paths = []
        for image_path in self.data_table.keys():
            self.image_paths.append(os.path.join(dataset_path, image_path))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        label_list = self.data_table[os.path.basename(image_path)]
        label_indices = torch.as_tensor(
            [self.label_table[label_text] for label_text in label_list],
            dtype=torch.long,
        )
        # one-hot encoding
        labels = torch.zeros(len(self.label_table))
        labels[label_indices] = 1

        if self.transform is not None:
            image = self.transform(image)

        return image, labels


class TestDataest(Dataset):
    def __init__(self, test_json_path, label_json_path):
        super(TestDataest, self).__init__()

        self.data_table = json.load(open(test_json_path, "r"))
        self.label_table = json.load(open(label_json_path, "r"))

    def __len__(self):
        return len(self.data_table)

    def __getitem__(self, index):
        label_list = self.data_table[index]
        label_indices = torch.as_tensor(
            [self.label_table[label_text] for label_text in label_list],
            dtype=torch.long,
        )
        # one-hot encoding
        labels = torch.zeros(len(self.label_table))
        labels[label_indices] = 1

        return labels


# if __name__ == "__main__":
# transform = T.Compose(
#     [
#         T.ToTensor(),
#         T.Lambda(lambda x: x * 2.0 - 1.0),
#     ]
# )
# dataset = IclevrDataset(
#     dataset_path="iclevr",
#     data_json_path="train.json",
#     label_json_path="objects.json",
#     transform=transform,
# )
# print(dataset[15])

# test_dataset = TestDataest(
#     test_json_path="test.json", label_json_path="objects.json"
# )
# print(test_dataset[0])
