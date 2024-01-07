from typing import Optional
import os
import warnings

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from . import transforms

warnings.simplefilter("ignore")


class SegmentationDataset(Dataset):

    def __init__(
            self,
            images_dir: str,
            masks_dir: str,
            ids: Optional[list] = None,
            transform_name: Optional[str] = None,
    ):
        super().__init__()
        self.ids = ids if ids is not None else os.listdir(images_dir)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transforms.__dict__[transform_name] if transform_name else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id = self.ids[i]
        image_path = os.path.join(self.images_dir, id)
        mask_path = os.path.join(self.masks_dir, id)

        # read data sample
        sample = dict(
            id=id,
            image=self.read_image(image_path),
            mask=self.read_mask(mask_path),
        )

        # apply augmentations
        if self.transform is not None:
            sample = self.transform(**sample)

        sample["mask"] = sample["mask"][None]  # expand first dim for mask

        return sample

    def read_image(self, path):
        image = Image.open(path)
        return np.array(image)

    def read_mask(self, path):
        image = self.read_image(path)
        return image/255