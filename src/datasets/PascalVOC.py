from typing import Optional
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

from .utils import download_and_extract_archive
from . import transforms

_DATASET_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
_SHA1_HASH = "4e443f8a2eca6b1dac8a6c57641b67dd40621a49"
_FILENAME = "VOCtrainval_11-May-2012.tar"
_BASEDIR = os.path.join("VOCdevkit", "VOC2012")


# Helper function to convert from one-hot encoded network output to color
def VOC_onehot2Color(one_hot_encoded_mask):

    COLOR_MAP = {
        0: [0, 0, 0],
        1: [128, 0, 0],
        2: [0, 128, 0],
        3: [128, 128, 0],
        4: [0, 0, 128],
        5: [128, 0, 128],
        6: [0, 128, 128],
        7: [128, 128, 128],
        8: [64, 0, 0],
        9: [192, 0, 0],
        10: [64, 128, 0],
        11: [192, 128, 0],
        12: [64, 0, 128],
        13: [192, 0, 128],
        14: [64, 128, 128],
        15: [192, 128, 128],
        16: [0, 64, 0],
        17: [128, 64, 0],
        18: [0, 192, 0],
        19: [128, 192, 0],
        20: [0, 64, 128],
    }

    assert_msg = 'Input one hot encoded mask shall be a HxWxN_Classes ndarray'
    assert isinstance(one_hot_encoded_mask, np.ndarray), assert_msg
    assert len(one_hot_encoded_mask.shape) == 3, assert_msg
    assert one_hot_encoded_mask.shape[2] == len(COLOR_MAP), assert_msg

    integer_mask = np.argmax(one_hot_encoded_mask, axis=2)

    # Initialize an empty RGB mask
    height, width = integer_mask.shape
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Map class indices to RGB colors
    for class_idx, color in COLOR_MAP.items():
        rgb_mask[integer_mask == class_idx] = color 

    return rgb_mask

class VOCSegmentationDataset(Dataset):

    def __init__(
            self,
            dst_root: str,
            image_set: str = "train",
            ids: Optional[list] = None,
            transform_name: Optional[str] = None,
            download : Optional[bool] = False,
    ):
        super().__init__()

        voc_root = os.path.join(dst_root, _BASEDIR)

        self.transform = transforms.__dict__[transform_name] if transform_name else None

        valid_image_sets = ["train", "trainval", "val"]

        if not image_set in valid_image_sets:
            raise RuntimeError("Dataset set not selected correctly, valid options: train, trainval, val")

        if download:
            download_and_extract_archive(_DATASET_URL, dst_root, voc_root ,filename=_FILENAME, md5=_SHA1_HASH)

        if not os.path.isdir(voc_root):
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
        
        splits_dir = os.path.join(voc_root, "ImageSets", "Segmentation")
        split_sel = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")

        with open(os.path.join(split_sel)) as f:
            file_names = [x.strip() for x in f.readlines()]

        image_dir = os.path.join(voc_root, "JPEGImages")
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]

        target_dir = os.path.join(voc_root, "SegmentationClass")
        self.targets = [os.path.join(target_dir, x + ".png") for x in file_names]

        assert len(self.images) == len(self.targets)

        self.color_map = {
                            0: [0, 0, 0],
                            1: [128, 0, 0],
                            2: [0, 128, 0],
                            3: [128, 128, 0],
                            4: [0, 0, 128],
                            5: [128, 0, 128],
                            6: [0, 128, 128],
                            7: [128, 128, 128],
                            8: [64, 0, 0],
                            9: [192, 0, 0],
                            10: [64, 128, 0],
                            11: [192, 128, 0],
                            12: [64, 0, 128],
                            13: [192, 0, 128],
                            14: [64, 128, 128],
                            15: [192, 128, 128],
                            16: [0, 64, 0],
                            17: [128, 64, 0],
                            18: [0, 192, 0],
                            19: [128, 192, 0],
                            20: [0, 64, 128],
        }

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i):

        # read data sample
        # We add image name as id
        sample = dict(
            id=self.images[i],
            image=self.read_image(self.images[i]),
            mask=self.read_mask(self.targets[i]),
        )

        # apply augmentations
        # We cannot convert from HWC to CHW in this step
        if self.transform is not None:
            sample = self.transform(**sample)

        # That transformation HWC to CHW should be applied in this step
        # in order to convert mask to one-hot encoding sucessfully
        sample["mask"] = self.onehot_encode(sample["mask"])

        return sample
    
    def onehot_encode(self, voc_color_mask):
        # If we are going to train semantic segmentation model,
        # the number of chanels of the mask should equal number
        # of classes and with 1's in the channel that the class
        # belongs
        # Create one-hot encoded mask
        num_classes = len(self.color_map)
        one_hot_mask = torch.zeros((voc_color_mask.shape[0], voc_color_mask.shape[1], num_classes))

        for class_idx, color in self.color_map.items():
            mask = np.all(voc_color_mask == color, axis=2)
            one_hot_mask[:, :, class_idx] = torch.tensor(mask, dtype=torch.float32)

        # HWC to CHW
        one_hot_mask = one_hot_mask.permute(2, 0, 1)
        return one_hot_mask

    def read_image(self, path):
        image = Image.open(path).convert('RGB')
        return np.array(image)

    def read_mask(self, path):
        mask = self.read_image(path)
        return mask
    
