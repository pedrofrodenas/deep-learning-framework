import collections
from typing import Optional, Any, Dict
import os
from xml.etree.ElementTree import Element
try:
    from defusedxml.ElementTree import parse
except ImportError:
    from xml.etree.ElementTree import parse

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

DATASET_YEAR_DICT = {
    "2012": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        "filename": "VOCtrainval_11-May-2012.tar",
        "sha1_hash": "6cd6e144f989b92b3379bac3b3de84fd",
        "base_dir": os.path.join("VOCdevkit", "VOC2012"),
    },
    "2007": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "filename": "VOCtrainval_06-Nov-2007.tar",
        "sha1_hash": "34ed68851bce2a36e2a223fa52c661d592c66b3c",
        "base_dir": os.path.join("VOCdevkit", "VOC2007"),
    },
    "2007-test": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
        "filename": "VOCtest_06-Nov-2007.tar",
        "md5": "41a8d6e12baa5ab18ee7f8f8029b9e11805b4ef1",
        "base_dir": os.path.join("VOCdevkit", "VOC2007"),
    },
}

class VOCDataset(Dataset):
    def __init__(self,
                 dst_root: str,
                 year: str,
                 image_set: str,
                 download: bool = False,
                 transform_name: Optional[str] = None,):

        if not year in ["2007", "2012"]:
            raise RuntimeError("Dataset set not selected correctly, valid options: 2007 or 2013")
        
        valid_image_sets = ["train", "trainval", "val"]

        if year == "2007":
            valid_image_sets.append("test")

        if not image_set in valid_image_sets:
            raise RuntimeError("Dataset set not selected correctly, valid options: train, trainval, val")
        
        key = "2007-test" if year == "2007" and image_set == "test" else year
        dataset_year_dict = DATASET_YEAR_DICT[key]

        voc_root = os.path.join(dst_root, dataset_year_dict['base_dir'])

        if download:
            download_and_extract_archive(dataset_year_dict['url'], 
                                         dst_root, voc_root ,
                                         filename=dataset_year_dict['filename'], 
                                         md5=dataset_year_dict['sha1_hash'])
            
        if not os.path.isdir(voc_root):
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
        
        # Get the file names (images and labels) for selected task
        task_dir = os.path.join(voc_root, "ImageSets", self._TASK_DIR)
        split_sel = os.path.join(task_dir, image_set.rstrip("\n") + ".txt")

        with open(os.path.join(split_sel)) as f:
            file_names = [x.strip() for x in f.readlines()]

        # Get image paths
        image_dir = os.path.join(voc_root, "JPEGImages")
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]

        # Get labels paths
        target_dir = os.path.join(voc_root, self._TARGET_DIR)
        self.targets = [os.path.join(target_dir, x + self._TARGET_FILE_EXT) for x in file_names]

        assert len(self.images) == len(self.targets)

    def __len__(self) -> int:
        return len(self.images)
    
class VOCSegmentation(VOCDataset):

    _TASK_DIR = "Segmentation"
    _TARGET_DIR = "SegmentationClass"
    _TARGET_FILE_EXT = ".png"

    def __init__(
            self,
            dst_root: str,
            year: str,
            image_set: str = "train",
            ids: Optional[list] = None,
            transform_name: Optional[str] = None,
            download : Optional[bool] = False,
            one_hot_encode : Optional[bool] = False,
    ):
        super().__init__(dst_root=dst_root, year=year, 
                         image_set=image_set, download=download)

        self.one_hot_encode = one_hot_encode
        self.transform = transforms.__dict__[transform_name] if transform_name else None

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

        if self.one_hot_encode:
            # That transformation HWC to CHW should be applied in this step
            # in order to convert mask to one-hot encoding sucessfully
            sample["mask"] = self.onehot_encode(sample["mask"]).long()
        else:
            sample["mask"] = self.index_encode(sample["mask"]).long()

        return sample
    
    def index_encode(self, voc_color_mask):
        # In pytorch CrossEntropyLoss does't need a one-hoy
        # encoded mask only an integer indicating the class type
        # In tensorflow masks in semantic segmentation should be
        # codified as one-hot encoded, in Pytorch is not neccessary
        # because CrossEntropyLoss is different from CategoricalCrossEntropy
        # from tensorflow
        num_classes = len(self.color_map)
        integer_mask = torch.zeros((voc_color_mask.shape[0], voc_color_mask.shape[1]))
        for class_idx, color in self.color_map.items():
            mask = np.all(voc_color_mask == color, axis=2)
            mask_int = mask.astype('uint8') * class_idx
            integer_mask += mask_int
        
        return integer_mask

    
    def onehot_encode(self, voc_color_mask):
        # If we are going to train semantic segmentation model,
        # the number of chanels of the mask should equal number
        # of classes and with 1's in the channel that the class
        # belongs [Tensorflow Format]
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
    
class VOCDetection(VOCDataset):
    _TASK_DIR = "Main"
    _TARGET_DIR = "Annotations"
    _TARGET_FILE_EXT = ".xml"

    def __init__(
            self,
            dst_root: str,
            year: str,
            image_set: str = "train",
            ids: Optional[list] = None,
            transform_name: Optional[str] = None,
            download : Optional[bool] = False,
    ):
        super().__init__(dst_root=dst_root, year=year, 
                         image_set=image_set, download=download)

        self.transform = transforms.__dict__[transform_name] if transform_name else None

    def __getitem__(self, i):

        # read data sample
        # We add image name as id
        sample = dict(
            id=self.images[i],
            image=self.read_image(self.images[i]),
            bbox=self.parse_voc_xml(parse(self.targets[i]).getroot())
        )

        # apply augmentations
        # We cannot convert from HWC to CHW in this step
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample
    
    def read_image(self, path):
        image = Image.open(path).convert('RGB')
        return np.array(image)
    
    @staticmethod
    def parse_voc_xml(node: Element) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(VOCDetection.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

    

    
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