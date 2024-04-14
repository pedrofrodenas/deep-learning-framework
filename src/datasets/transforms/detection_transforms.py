import numpy as np
from typing import Any, Dict, Sequence, Tuple, Union
import random
import warnings

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
from albumentations.core.types import BoxInternalType, KeypointInternalType
import cv2
    

# --------------------------------------------------------------------
# Voc Detection transforms
# --------------------------------------------------------------------

voc_detection = A.Compose([
                    A.Resize(300, 300),
                    #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))