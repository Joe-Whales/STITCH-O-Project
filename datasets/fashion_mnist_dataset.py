from __future__ import division

import logging
import os.path
import pickle
import random
from typing import Any, List

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

logger = logging.getLogger("global_logger")

classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        
def build_fmnist_dataloader(cfg, training, distributed=True):
    logger.info("building CustomDataset from: {}".format(cfg["root_dir"]))
    transform = transforms.Compose([
                    transforms.Resize((cfg["input_size"][0], cfg["input_size"][1])),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),])    
    if training:
        dataset = FashionMNIST(root='./', train=True,download=True,
                           normal_set=cfg["normals"], transform=transform)
    else:
        dataset = FashionMNIST(root='./', train=False, download=True,
                           normal_set=cfg["normals"], transform=transform)
    sampler = RandomSampler(dataset)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        sampler=sampler,
    )
    return data_loader

import torchvision

class FashionMNIST(torchvision.datasets.MNIST):
    
    mirrors = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]

    resources = [
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
    ]
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    def __init__(self,root: str, train: bool = True, normal_set: list=[0], transform= None,
        target_transform = None, download: bool = False):
        super().__init__(root, transform=transform, download=download, target_transform=target_transform)
        self.train = train  # training 

        if self._check_legacy_exist():
            self.data, self.targets = self._load_legacy_data()
            return
        download = True
        if download:
            self.download()
        # if not self._check_exists():
            # raise RuntimeError("Dataset not found. You can use download=True to download it")
        self.data, self.targets = self._load_data()
        normal_indice=[]
        anomaly_indice=[]
        for k in range(len(self.targets)):
            if self.targets[k] in normal_set:
                normal_indice.append(k)
            else:
                anomaly_indice.append(k)
        self.targets[normal_indice] = 0
        self.targets[anomaly_indice] = 1
        if train:
            self.targets = self.targets[normal_indice]
            self.data = self.data[normal_indice]

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")
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