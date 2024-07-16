from __future__ import division

import logging
import os.path
import pickle
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg


logger = logging.getLogger("global_logger")

classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        
def build_svhn_dataloader(cfg, training, distributed=True):
    logger.info("building CustomDataset from: {}".format(cfg["root_dir"]))
    transform = transforms.Compose([
                    transforms.Resize((cfg["input_size"][0], cfg["input_size"][1])),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),])    
    if training:
        dataset = SVHN(root='./', normal_set=cfg["normals"], split="train", transform=transform, download=True)
    else:
        dataset = SVHN(root='./', normal_set=cfg["normals"], split="test", transform=transform, download=True)
    sampler = RandomSampler(dataset)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        sampler=sampler,
    )
    return data_loader


classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

class SVHN(VisionDataset):
    split_list = {
        "train": [
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
            "train_32x32.mat",
            "e26dedcc434d2e4c54c9b2d4a06d8373",
        ],
        "test": [
            "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
            "test_32x32.mat",
            "eb5a983be6a315427106f1b164d9cef3",
        ],
        "extra": [
            "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
            "extra_32x32.mat",
            "a93ce644f1a588dc4d68dda5feec44a7",
        ],
    }

    def __init__(
        self,
        root: str,
        normal_set: list = [1],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]
        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
        import scipy.io as sio
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))
        self.data = loaded_mat["X"]
        self.labels = loaded_mat["y"].astype(np.int64).squeeze()
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

        normal_indice=[]
        anomaly_indice=[]
        for k in range(len(self.labels)):
            if self.labels[k] in normal_set:
                normal_indice.append(k)
            else:
                anomaly_indice.append(k)
        self.labels[normal_indice] = 0
        self.labels[anomaly_indice] = 1
        if split=='train':
            self.labels = self.labels[normal_indice]
            self.data = self.data[normal_indice]
        if split=='test':
            ids = random.sample(range(len(self.labels)), 15000)
            self.labels = self.labels[ids]
            self.data = self.data[ids]


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.labels[index])
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        height = img.shape[1]
        width = img.shape[2]
        if target == 0:
            mask = torch.zeros((1, height, width))
        else:
            mask = torch.ones((1, height, width))

        input = {
            "filename": "{}/{}.jpg".format(classes[target], index),
            "image": img,
            "mask": mask,
            "height": height,
            "width": width,
            "label": target,
            "clsname": "fashion-mnist",
        }
        return input   

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)
