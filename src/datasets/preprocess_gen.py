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


def parse_defaultannot(defaultannot_path):
    
    defaultannot = {}
    
    with open(defaultannot_path) as f:
        for line in f:
            line_list = line.split()
            integer_map = list(map(int, line_list[:-1]))
            label_name = line_list[-1]
            defaultannot[label_name] = integer_map
    return defaultannot

def filter_defects(img, filter_names, defaultannot):
    
    img = np.array(img)
    
    for defect_name in filter_names:
        
        rgb_code = defaultannot[defect_name]
        
        mask = np.all(img == rgb_code, axis=2)
        
        img[mask, :] = [0, 0, 0]
        
    return img

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

        mask.point(lambda p: p > 0 and 255)

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
    
    return args_parser.parse_args()


      
def main():
    
    args = get_args()

    extension = args.extension

    
    image_paths = glob.glob(args.image_folder + "/*" + extension)

    dst_defect_image = os.path.join(args.dst_folder, 'images')
    dst_defect_mask = os.path.join(args.dst_folder, 'masks')
  
    if not os.path.exists(dst_defect_image):
        os.makedirs(dst_defect_image)
    
    if not os.path.exists(dst_defect_mask):
        os.makedirs(dst_defect_mask)
        
  
    processor = Preprocessor(args.ground_truth,
                             dst_defect_image, 
                             dst_defect_mask,
                             [512, 512])
    
    _parallelize(processor.preprocess_image, image_paths)
    
    
if __name__ == '__main__':
    main()