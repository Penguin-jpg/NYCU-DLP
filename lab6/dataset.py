from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import json

PAD = 24


class IclevrDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        data_json_path,
        label_json_path,
        use_multi_hot=True,
        transform=None,
    ):
        super(IclevrDataset, self).__init__()

        self.data_table = json.load(open(data_json_path, "r"))
        self.label_table = json.load(open(label_json_path, "r"))
        self.use_multi_hot = use_multi_hot
        self.max_label_length = max(
            [len(label_list) for label_list in self.data_table.values()]
        )

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
        if self.transform is not None:
            image = self.transform(image)

        label_list = self.data_table[os.path.basename(image_path)]
        label_indices = torch.as_tensor(
            [self.label_table[label_text] for label_text in label_list],
            dtype=torch.long,
        )

        if self.use_multi_hot:
            # multi-hot encoding
            labels = torch.zeros(len(self.label_table))
            labels[label_indices] = 1
            return image, labels

        # pad to max_label_length
        label_indices = F.pad(
            label_indices,
            (0, self.max_label_length - len(label_indices)),
            "constant",
            PAD,
        )
        return image, label_indices


class TestDataest(Dataset):
    def __init__(self, test_json_path, label_json_path, use_multi_hot=True):
        super(TestDataest, self).__init__()

        self.data_table = json.load(open(test_json_path, "r"))
        self.label_table = json.load(open(label_json_path, "r"))
        self.use_multi_hot = use_multi_hot
        self.max_label_length = max([len(label_list) for label_list in self.data_table])

    def __len__(self):
        return len(self.data_table)

    def __getitem__(self, index):
        label_list = self.data_table[index]
        label_indices = torch.as_tensor(
            [self.label_table[label_text] for label_text in label_list],
            dtype=torch.long,
        )

        if self.use_multi_hot:
            # multi-hot encoding
            labels = torch.zeros(len(self.label_table))
            labels[label_indices] = 1
            return labels

        # pad to max_label_length
        label_indices = F.pad(
            label_indices,
            (0, self.max_label_length - len(label_indices)),
            "constant",
            PAD,
        )
        return label_indices


# if __name__ == "__main__":
#     transform = T.Compose(
#         [
#             T.ToTensor(),
#             T.Lambda(lambda x: x * 2.0 - 1.0),
#         ]
#     )
# dataset = IclevrDataset(
#     dataset_path="iclevr",
#     data_json_path="train.json",
#     label_json_path="objects.json",
#     use_multi_hot=True,
#     transform=transform,
# )
# print(dataset[15])
# dataset = IclevrDataset(
#     dataset_path="iclevr",
#     data_json_path="train.json",
#     label_json_path="objects.json",
#     use_multi_hot=False,
#     transform=transform,
# )
# print(dataset[15])

# test_dataset = TestDataest(
#     test_json_path="test.json", label_json_path="objects.json", use_multi_hot=True
# )
# print(test_dataset[0])

# test_dataset = TestDataest(
#     test_json_path="test.json", label_json_path="objects.json", use_multi_hot=False
# )
# print(test_dataset[0])
