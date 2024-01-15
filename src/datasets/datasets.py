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
    
class PredictionDataset(Dataset):

    def __init__(
            self,
            images_dir: str,
            shape: [tuple],
            ids: Optional[list] = None,
            transform_name: Optional[str] = None,
    ):
        super().__init__()
        self.ids = ids if ids is not None else os.listdir(images_dir)
        self.images_dir = images_dir
        self.transform = transforms.__dict__[transform_name] if transform_name else None
        self.shape = shape

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id = self.ids[i]
        image_path = os.path.join(self.images_dir, id)

        image = self.read_image(image_path)

        padded_image, left, top, r = self.preprocess_image(image)
        # read data sample
        sample = dict(
            id=id,
            image=np.array(padded_image),
            left = left,
            top = top,
            r = r
        )
        # apply augmentations
        if self.transform is not None:
            sample = self.transform(**sample)
            
        return sample

    def read_image(self, path):
        image = Image.open(path)
        return image
    
    def preprocess_image(self, image):
        
        scaleup = True
    
        if image.mode != 'RGB':
            image = image.convert('RGB')

         # Resize and pad image while meeting stride-multiple constraints
        shape = image.size  # current shape [width, height]

        # Scale ratio (new / old)
        r = min(self.shape[1] / shape[0], self.shape[0] / shape[1])

        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))

        dw, dh = self.shape[1] - new_unpad[0], self.shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            image = image.resize(new_unpad, Image.BILINEAR)

        padded_img = Image.new(image.mode, (self.shape[1], self.shape[0]), 0)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        padded_img.paste(image, (left, top))
       
        return padded_img, left, top, r