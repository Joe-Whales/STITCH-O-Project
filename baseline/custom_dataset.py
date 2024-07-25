import json
import logging

# Ensure the root directory is in the PYTHONPATH
import sys
import os

# Add the root directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch import from_numpy
from torch.nn import Module as nn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
import matplotlib.pyplot as plt
from torchvision import transforms

logger = logging.getLogger("global_logger")


def build_custom_dataloader(cfg, training, distributed=True):

    transform_fn = transforms.Resize(cfg["input_size"])

    logger.info("building CustomDataset from: {}".format(cfg["meta_file"]))

    dataset = CustomDataset(
        cfg["meta_file"],
        training,
        image_path=cfg["image_path"],
        transform_fn=transform_fn,
    )

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
        filename = os.path.join(self.image_path, filename)
        image = np.load(filename)

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
        image = from_numpy(image).float().permute(2, 0, 1)

        mask = Image.fromarray(mask, "L")

        if self.transform_fn:
            image = self.transform_fn(image)
            mask = self.transform_fn(mask)

        mask = transforms.ToTensor()(mask)
        # else:
        normalize_fn = transforms.Normalize(mean=[0.485], std=[0.229])
        
        image = normalize_fn(image)
        
        # duplicate channels of image to 3
        if image.size(0) == 1:
            image = image.expand(3, -1, -1)
        
        input.update({"image": image, "mask": mask})

        return input