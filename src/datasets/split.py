import argparse
import glob
import random
import os
import shutil

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
        '--dst-train',
        help='path to dst output train images',
        type=str,
        required=True)
    
    args_parser.add_argument(
        '--dst-val',
        help='path to dst output val images',
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

    # Shuffle input image paths
    random.shuffle(image_paths)

    dst_image_train = os.path.join(args.dst_train, 'images')
    dst_mask_train = os.path.join(args.dst_train, 'masks')


    dst_image_val = os.path.join(args.dst_val, 'images')
    dst_mask_val = os.path.join(args.dst_val, 'masks')
  
    if not os.path.exists(dst_image_train):
        os.makedirs(dst_image_train)
    
    if not os.path.exists(dst_mask_train):
        os.makedirs(dst_mask_train)

    if not os.path.exists(dst_image_val):
        os.makedirs(dst_image_val)

    if not os.path.exists(dst_mask_val):
        os.makedirs(dst_mask_val)

    n_test_images = int(len(image_paths)*0.3)

    test_images = image_paths[:n_test_images]
    train_images = image_paths[n_test_images:]

    for train_image in train_images:

        file_name = os.path.basename(train_image)
        shutil.copy(train_image, dst_image_train)
        shutil.copy(os.path.join(args.ground_truth, file_name), dst_mask_train)

    
    for test_image in test_images:

        file_name = os.path.basename(test_image)
        shutil.copy(test_image, dst_image_val)
        shutil.copy(os.path.join(args.ground_truth, file_name), dst_mask_val)

    

    


    
    
if __name__ == '__main__':
    main()