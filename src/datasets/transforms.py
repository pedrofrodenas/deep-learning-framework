import numpy as np
from typing import Any, Dict, Sequence, Tuple, Union
import random
import warnings

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
from albumentations.core.types import BoxInternalType, KeypointInternalType
import cv2

warnings.simplefilter("ignore")

# --------------------------------------------------------------------
# Helpful functions
# --------------------------------------------------------------------

def post_transform(image, **kwargs):
    if image.ndim == 3:
        return image.transpose(2, 0, 1).astype("float32")
    else:
        return image.astype("float32")
    
def full_post_transform(image ,**kwargs):
    if image.ndim == 4:
        return image.transpose(0,3, 1, 2).astype("float32")
    elif image.ndim == 3:
        return image.astype("float32")
    
class ImageHWCtoCHW(ImageOnlyTransform):
    """
    RamdomA transformation
    """
    def __init__(self) -> None:
        super(ImageHWCtoCHW, self).__init__(p=1)

    def apply(self, image, **kwargs):
        if image.ndim == 3:
            return image.transpose(2, 0, 1).astype("float32")
        else:
            return image.astype("float32")
        
class Threshold(DualTransform):
    """Apply threshold to mask without modifying the input image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        threshold: float = 0.5,
        always_apply: bool = False,
        p: float = 1,
    ):
        super().__init__(always_apply, p)
        self.threshold = threshold

    def apply(
        self, img: np.ndarray, **params: Any
    ) -> np.ndarray:
        return img
    
    def apply_to_mask(self, mask: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        return (mask >= 0.5).astype('float64')
        
class LongestMaxSizeCustom(DualTransform):
    """Rescale an image so that maximum side is equal to max_size, keeping the aspect ratio of the initial image.

    Args:
        max_size (int, list of int): maximum size of the image after the transformation. When using a list, max size
            will be randomly selected from the values in the list.
        interpolation (OpenCV flag): interpolation method. Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        max_size: Union[int, Sequence[int]] = 1024,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool = False,
        p: float = 1,
    ):
        super().__init__(always_apply, p)
        self.interpolation = interpolation
        self.max_size = max_size

    def apply(
        self, img: np.ndarray, max_size: int = 1024, interpolation: int = cv2.INTER_LINEAR, **params: Any
    ) -> np.ndarray:
        return F.longest_max_size(img, max_size=max_size, interpolation=interpolation)

    def apply_to_bbox(self, bbox: BoxInternalType, **params: Any) -> BoxInternalType:
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(
        self, keypoint: KeypointInternalType, max_size: int = 1024, **params: Any
    ) -> KeypointInternalType:
        height = params["rows"]
        width = params["cols"]

        scale = max_size / max([height, width])
        return F.keypoint_scale(keypoint, scale, scale)
    
    def apply_to_mask(self, mask: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        return self.apply(mask, **{k: cv2.INTER_AREA if k == "interpolation" else v for k, v in params.items()})

    def get_params(self) -> Dict[str, int]:
        return {"max_size": self.max_size if isinstance(self.max_size, int) else random.choice(self.max_size)}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("max_size", "interpolation")
    
# --------------------------------------------------------------------
# VOC Dataset
# --------------------------------------------------------------------


VOC_train_transform = A.Compose(
    [
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.RandomCrop(256, 256),
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.05, rotate_limit=15, p=0.4),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ImageHWCtoCHW(),
    ]
)

VOC_val_transform = A.Compose([
    A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT),
    A.LongestMaxSize(max_size=256, interpolation=1, always_apply=False, p=1),
    A.RandomCrop(height=224, width=224, always_apply=False, p=1),
    ImageHWCtoCHW(),
])

# --------------------------------------------------------------------
# Customs transforms
# --------------------------------------------------------------------

custom_transform_v1 = A.Compose(
    [
        A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.05, rotate_limit=15, p=0.4),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.PadIfNeeded(min_height=410, min_width=410, border_mode=cv2.BORDER_CONSTANT),
        LongestMaxSizeCustom(max_size=410, interpolation=1, always_apply=False, p=1),
        A.PadIfNeeded(min_height=410, min_width=410, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.2),
        A.RandomCrop(106, 106, p=1),
        ImageHWCtoCHW(),
    ]
)

custom_transform_v2 = A.Compose([
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.HorizontalFlip(p=0.2),
    A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT),
    LongestMaxSizeCustom(max_size=224, interpolation=1, always_apply=False, p=1),
    A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT),
    ImageHWCtoCHW(),
])

# Combine the sets with equal probabilities
custom_train_transform = A.Compose([
    A.OneOf([custom_transform_v1, custom_transform_v2], p=1.0)
])

# Thresholding in this transform is needed depending on the loss used
custom_test_transform = A.Compose([A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT),
                                LongestMaxSizeCustom(max_size=224, interpolation=1, always_apply=False, p=1),
                                A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT), 
                                ImageHWCtoCHW()])

# --------------------------------------------------------------------
# Voc Detection transforms
# --------------------------------------------------------------------

voc_detection = A.Compose([
                    A.Resize(224, 224),
                    #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    #ToTensorV2()
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# --------------------------------------------------------------------
# Segmentation transforms
# --------------------------------------------------------------------

post_transform = A.Lambda(name="post_transform", image=post_transform, mask=post_transform)

full_post_transform = A.Lambda(name="full_post_transform", image=full_post_transform, mask=full_post_transform)

train_transform = A.Compose([
    A.OneOf([A.RandomSizedCrop(min_max_height=(150, 300), height=512, width=512, p=0.2),
             A.PadIfNeeded(min_height=512, min_width=512, p=0.5)], p=0.2),
    A.OneOf([A.RandomBrightnessContrast(p=0.1),
             A.CLAHE(p=0.3),
             A.GaussianBlur(3, p=0.3),
             A.Sharpen(alpha=(0.1, 0.22), p=0.3),
             A.RandomGamma(p=0.1)]),
    post_transform
    ])

# crop 512
train_transform_1 = A.Compose([
    A.RandomCrop(512, 512, p=1.),
    A.Flip(p=0.75),
    A.RandomBrightnessContrast(p=0.5),
    post_transform,
])

valid_transform_1 = A.Compose([
    post_transform,
])

test_transform_1 = A.Compose([
    post_transform,
])

valid_transform_2 = valid_transform_1
test_transform_2 = test_transform_1
