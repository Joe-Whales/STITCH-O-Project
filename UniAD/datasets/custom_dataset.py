from __future__ import division

import json
import logging

from torch import from_numpy, cat
from torch.nn import Module as nn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from datasets.base_dataset import BaseDataset, TestBaseTransform, TrainBaseTransform
from datasets.image_reader import build_image_reader
from datasets.transforms import RandomColorJitter
import matplotlib.pyplot as plt

logger = logging.getLogger("global_logger")


def build_custom_dataloader(cfg, training, distributed=True):

    image_reader = build_image_reader(cfg.image_reader)

    if training:
        transform_fn = TrainBaseTransform(
            cfg["input_size"], cfg["hflip"], cfg["vflip"], cfg["rotate"]
        )
    else:
        transform_fn = TestBaseTransform(cfg["input_size"])

    colorjitter_fn = None
    if cfg.get("colorjitter", None) and training:
        colorjitter_fn = RandomColorJitter.from_params(cfg["colorjitter"])

    logger.info("building CustomDataset from: {}".format(cfg["meta_file"]))

    dataset = CustomDataset(
        image_reader,
        cfg["meta_file"],
        training,
        transform_fn=transform_fn,
        colorjitter_fn=colorjitter_fn,
    )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

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


class CustomDataset(BaseDataset):
    def __init__(
        self,
        image_reader,
        meta_file,
        training,
        transform_fn,
        colorjitter_fn=None,
    ):
        self.image_reader = image_reader
        self.meta_file = meta_file
        self.training = training
        self.transform_fn = transform_fn
        self.colorjitter_fn = colorjitter_fn

        # construct metas
        with open(meta_file, "r") as f_r:
            self.metas = []
            for line in f_r:
                meta = json.loads(line)
                self.metas.append(meta)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        input = {}
        meta = self.metas[index]

        # read image
        filename = meta["filename"]
        label = meta["label"]
        image = self.image_reader(meta["filename"])

        input.update(
            {
                "filename": filename,
                "height": image.shape[0],
                "width": image.shape[1],
                "label": label,
            }
        )

        if meta.get("clsname", None):
            input["clsname"] = meta["clsname"]
        else:
            input["clsname"] = filename.split("/")[-4]

        # read / generate mask
        if meta.get("maskname", None):
            mask = self.image_reader(meta["maskname"], is_mask=True)
        else:
            if label == 0:  # good
                mask = np.zeros((input["height"], input["width"])).astype(np.uint8)
            elif label == 1:  # defective
                mask = (np.ones((input["height"], input["width"])) * 255).astype(np.uint8)
            else:
                raise ValueError("Labels must be [None, 0, 1]!")

        # convert image to tensor and permute
        image = from_numpy(image).float()

        # Check the number of channels
        if len(image.shape) == 2:  # Single channel
            image = image.unsqueeze(0)
        elif len(image.shape) == 3:
            image = image.permute(2, 0, 1)
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        mask = Image.fromarray(mask, "L")

        if self.transform_fn:
            image, mask = self.transform_fn(image, mask)
        if self.colorjitter_fn:
            image = self.colorjitter_fn(image)

        mask = transforms.ToTensor()(mask)
        normalize_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406][:image.size(0)], std=[0.229, 0.224, 0.225][:image.size(0)])

        image = normalize_fn(image)

        # Handle different channel counts
        if image.size(0) == 1:
            image = image.expand(3, -1, -1)
        elif image.size(0) == 2:
            image = cat([image, image[0].unsqueeze(0)], dim=0)
        elif image.size(0) > 3:
            image = image[:3]

        input.update({"image": image, "mask": mask})

        return input