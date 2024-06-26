#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 13:55:07 2022

@author: pedrofRodenas
"""

import argparse
import glob
import math
from multiprocessing import Pool, cpu_count
import os
import shutil
import zipfile

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def numpy2pil(np_array: np.ndarray) -> Image:
    """
    Convert an HxWx3 numpy array into an RGB Image
    """

    assert_msg = 'Input shall be a HxWx3 ndarray'
    assert isinstance(np_array, np.ndarray), assert_msg
    assert len(np_array.shape) == 3, assert_msg
    assert np_array.shape[2] == 3, assert_msg

    img = Image.fromarray(np_array, 'RGB')
    return img

class PreprocessorOpenCV():
    
    def __init__(self,
                 ground_truth,
                 dst_defect_image, 
                 dst_defect_mask,
                 shape):
        
        self.ground_truth = ground_truth
        self.dst_defect_image = dst_defect_image
        self.dst_defect_mask = dst_defect_mask
        # shape [height, width]
        self.shape = shape
        
    def preprocess_image(self, image_path):
        
        scaleup = True
        image_name = os.path.splitext(os.path.basename(image_path))[0]+'.png'
        
        full_mask_path = os.path.join(self.ground_truth, image_name)
        
        image = cv2.imread(image_path)
        mask = cv2.imread(full_mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) 

        mask = (mask > 0).astype('uint8')*255


        # Resize and pad image while meeting stride-multiple constraints
        shape = image.shape  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(self.shape[0] / shape[0], self.shape[1] / shape[1])

        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        # new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
        new_unpad = int(math.floor(shape[0] * r)), int(math.floor(shape[1] * r))

        dw, dh = self.shape[1] - new_unpad[1], self.shape[0] - new_unpad[0]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[:2] != new_unpad:  # resize

            image = cv2.resize(image,
                    (int(shape[1] * r), int(shape[0] * r)),
                    interpolation = cv2.INTER_LINEAR
                )
            mask = cv2.resize(mask,
                    (int(shape[1] * r), int(shape[0] * r)),
                    interpolation = cv2.INTER_NEAREST
                )

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        image = cv2.copyMakeBorder(image, top, bottom, left, right, 0)
        mask = cv2.copyMakeBorder(mask, top, bottom, left, right, 0)

        cv2.imwrite(os.path.join(self.dst_defect_image, image_name), image)
        cv2.imwrite(os.path.join(self.dst_defect_mask, image_name), mask)

        
def _parallelize(func, data):
    processes = cpu_count() - 1
    with Pool(processes) as pool:
        # We need the enclosing list statement to wait for the iterator to end
        # https://stackoverflow.com/a/45276885/1663506
        list(tqdm(pool.imap_unordered(func, data), total=len(data)))

class Preprocessor():
    
    def __init__(self,
                 ground_truth,
                 dst_defect_image, 
                 dst_defect_mask,
                 shape):
        
        self.ground_truth = ground_truth
        self.dst_defect_image = dst_defect_image
        self.dst_defect_mask = dst_defect_mask
        # shape [height, width]
        self.shape = shape
        
    def preprocess_image(self, image_path):

        scaleup = True
        image_name = os.path.splitext(os.path.basename(image_path))[0]+'.png'
        
        full_mask_path = os.path.join(self.ground_truth, image_name)
        
        image = Image.open(image_path)
        mask = Image.open(full_mask_path).convert('L')

        mask = mask.point(lambda p: p > 0 and 255)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize and pad image while meeting stride-multiple constraints
        shape = image.size  # current shape [width, height]

        # Scale ratio (new / old)
        r = min(self.shape[0] / shape[0], self.shape[1] / shape[1])

        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))

        dw, dh = self.shape[1] - new_unpad[0], self.shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            image = image.resize(new_unpad, Image.BILINEAR)
            mask = mask.resize(new_unpad, Image.NEAREST)

        padded_img = Image.new(image.mode, (self.shape[1], self.shape[0]), 0)
        padded_mask = Image.new(mask.mode, (self.shape[1], self.shape[0]), 0)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        padded_img.paste(image, (left, top))
        padded_mask.paste(mask, (left, top))
       
        padded_img.save(os.path.join(self.dst_defect_image, image_name))
        padded_mask.save(os.path.join(self.dst_defect_mask, image_name))

        
def _parallelize(func, data):
    processes = cpu_count() - 1
    with Pool(processes) as pool:
        # We need the enclosing list statement to wait for the iterator to end
        # https://stackoverflow.com/a/45276885/1663506
        list(tqdm(pool.imap_unordered(func, data), total=len(data)))
    
    
    

def get_args():
    """Define the task arguments with the default values.
    Returns:
        experiment parameters
    """

    args_parser = argparse.ArgumentParser()
    
    args_parser.add_argument(
        '--image-folder',
        help='path to uncompressed images',
        type=str,
        required=True)
    
    args_parser.add_argument(
        '--ground-truth',
        help='path to uncompressed ground truth',
        type=str,
        required=True)
    
    args_parser.add_argument(
        '--dst-folder',
        help='path to output dir images with defects',
        type=str,
        required=True)
    
    args_parser.add_argument(
        '--extension',
        help='path to output dir where images with discarted defects are placed',
        type=str,
        required=True)
    
    args_parser.add_argument(
        '--input-size',
        help='input size of the dataset [height, width]',
        nargs='+',
        type=int,
        required=True)
    
    return args_parser.parse_args()


      
def main():
    
    args = get_args()

    extension = args.extension

    input_size = args.input_size

    
    image_paths = glob.glob(args.image_folder + "/*" + extension)

    dst_defect_image = os.path.join(args.dst_folder, 'images')
    dst_defect_mask = os.path.join(args.dst_folder, 'masks')
  
    if not os.path.exists(dst_defect_image):
        os.makedirs(dst_defect_image)
    
    if not os.path.exists(dst_defect_mask):
        os.makedirs(dst_defect_mask)
        
  
    processor = PreprocessorOpenCV(args.ground_truth,
                             dst_defect_image, 
                             dst_defect_mask,
                             input_size)
    
    _parallelize(processor.preprocess_image, image_paths)
    
    
if __name__ == '__main__':
    main()