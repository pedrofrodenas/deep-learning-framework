from typing import Optional
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .utils import download_and_extract_archive
from . import transforms

_DATASET_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
_SHA1_HASH = "4e443f8a2eca6b1dac8a6c57641b67dd40621a49"
_FILENAME = "VOCtrainval_11-May-2012.tar"
_BASEDIR = os.path.join("VOCdevkit", "VOC2012")



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

        self.color_map = [
                            [0, 0, 0],
                            [128, 0, 0],
                            [0, 128, 0],
                            [128, 128, 0],
                            [0, 0, 128],
                            [128, 0, 128],
                            [0, 128, 128],
                            [128, 128, 128],
                            [64, 0, 0],
                            [192, 0, 0],
                            [64, 128, 0],
                            [192, 128, 0],
                            [64, 0, 128],
                            [192, 0, 128],
                            [64, 128, 128],
                            [192, 128, 128],
                            [0, 64, 0],
                            [128, 64, 0],
                            [0, 192, 0],
                            [128, 192, 0],
                            [0, 64, 128],
                        ]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i):

        # read data sample
        sample = dict(
            id=id,
            image=self.read_image(self.images[i]),
            mask=self.read_mask(self.targets[i]),
        )

        # apply augmentations
        if self.transform is not None:
            sample = self.transform(**sample)

        sample["mask"] = sample["mask"][None]  # expand first dim for mask

        return sample
    
    def onehot_encode(self, mask):
        # If we are going to train semantic segmentation model,
        # the number of chanels of the mask should equal number
        # of classes and with 1's in the channel that the class
        # belongs
        height, width = mask.shape[:2]
        semantic_mask = np.zeros((height, width, len(self.color_map)), dtype=np.float32)
        for label_index, label in enumerate(self.color_map):
            semantic_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
        return semantic_mask

    def read_image(self, path):
        image = Image.open(path).convert('RGB')
        return np.array(image)

    def read_mask(self, path):
        mask = self.read_image(path)
        mask = self.onehot_encode(mask)
        return mask
    
