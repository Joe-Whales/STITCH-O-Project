# custom_dataset.py

import json
import logging
import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logger = logging.getLogger("global_logger")

def build_custom_dataloader(cfg, training, distributed=False):
    transform_fn = transforms.Resize(cfg["input_size"])

    logger.info("building CustomDataset from: {}".format(cfg["meta_file"]))

    dataset = CustomDataset(
        cfg["meta_file"],
        training,
        image_path=cfg["image_path"],
        transform_fn=transform_fn,
    )

    sampler = torch.utils.data.RandomSampler(dataset)

    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        sampler=sampler,
        prefetch_factor=2,
        persistent_workers=True,
    )

    return data_loader

class CustomDataset(Dataset):
    def __init__(
        self,
        meta_file,
        training,
        transform_fn,
        image_path,
    ):
        self.meta_file = meta_file
        self.training = training
        self.transform_fn = transform_fn
        self.image_path = image_path

        with open(meta_file, "r") as f_r:
            self.metas = [json.loads(line) for line in f_r]

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        input = {}
        meta = self.metas[index]

        filename = meta["filename"]
        label = meta["label"]
        filename = os.path.join(self.image_path, filename)
        
        # Load the image with both channels
        image = np.load(filename)
        
        # Ensure the image has 2 channels
        if image.shape[-1] != 2:
            raise ValueError(f"Expected 2 channels, but got {image.shape[-1]} for file {filename}")

        input.update(
            {
                "filename": filename,
                "height": image.shape[0],
                "width": image.shape[1],
                "label": label,
            }
        )

        input["clsname"] = meta.get("clsname", filename.split("/")[-4])

        # Split the channels
        normal_image = image[..., 0]
        abnormal_image = image[..., 1]

        # Convert images to tensors and add channel dimension
        normal_image = torch.from_numpy(normal_image).float().unsqueeze(0)
        abnormal_image = torch.from_numpy(abnormal_image).float().unsqueeze(0)

        if self.transform_fn:
            normal_image = self.transform_fn(normal_image)
            abnormal_image = self.transform_fn(abnormal_image)

        normalize_fn = transforms.Normalize(mean=[0.485], std=[0.229])
        
        normal_image = normalize_fn(normal_image)
        abnormal_image = normalize_fn(abnormal_image)
        
        # Duplicate channels to 3 if necessary
        if normal_image.size(0) == 1:
            normal_image = normal_image.expand(3, -1, -1)
        if abnormal_image.size(0) == 1:
            abnormal_image = abnormal_image.expand(3, -1, -1)
        
        input.update({"normal_image": normal_image, "abnormal_image": abnormal_image})

        return input