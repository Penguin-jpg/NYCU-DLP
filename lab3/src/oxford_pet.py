import os
import random
import shutil
from urllib.request import urlretrieve

import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import pad_image


class OxfordPetDataset(Dataset):
    def __init__(self, root, mode="train", pad=True):
        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.pad = pad

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")
        image = Image.open(image_path).convert("RGB")
        trimap = np.array(Image.open(mask_path))
        mask = Image.fromarray(self._preprocess_mask(trimap))
        if self.pad:
            image = pad_image(image)
            mask = pad_image(mask)

        if self.mode == "train" or self.mode == "valid":
            image, mask = self.transform(image, mask, flip=True)
        else:
            image, mask = self.transform(image, mask, flip=False)

        return image, mask

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):
        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

    # reference: https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/7
    def transform(self, image, mask, flip=True):
        # because we want identical transformation for both image and mask, I
        # cannot directly apply transformation that depends on probability on
        # both of them separately

        # resize doesn't depend on probability
        image = TF.resize(image, (256, 256))
        mask = TF.resize(mask, (256, 256))

        # random horizontal flip (anything mentioned random related to probability)
        if flip and random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # convert to tensor and normalize to [0, 1]
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # standardization (mean=0, std=1) to speed up convergence
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image, mask


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)


def load_dataset(data_path, mode, pad=True):
    return OxfordPetDataset(data_path, mode, pad)
