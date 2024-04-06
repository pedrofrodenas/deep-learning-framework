import os
import numpy as np
import cv2

class VOCInterpreter():
    def __init__(self, 
                 log_dir = None,
                 save_results=False, 
                 **kwargs):
        self.log_dir = log_dir
        self.save_results = save_results
        self.dst_dir = os.path.join(log_dir, "inference")

        if save_results:
            if not os.path.exists(self.dst_dir):
                os.makedirs(self.dst_dir)

    def __call__(self, dataloader_output, predictions):

        # Turn channel dim to last dimension and convert to numpy
        one_hot_mask = predictions['mask'].permute(0, 2, 3, 1).cpu().numpy()
        original_images = dataloader_output['image'].permute(0, 2, 3, 1).cpu().numpy()
        ids = dataloader_output['id']

        for id, single_image, single_mask in zip(ids, original_images, one_hot_mask):
            # Process single batch of data
            rgb_mask = self.VOC_onehot2Color(single_mask)
            rgb_img = single_image.astype('uint8')
            img_name = os.path.basename(id)

            if self.save_results:
                result = np.concatenate([rgb_img, rgb_mask], axis=1)
                result = cv2.cvtColor(result,  cv2.COLOR_RGB2BGR)
                file = os.path.join(self.dst_dir, img_name)
                cv2.imwrite(file, result)


    def VOC_onehot2Color(self, one_hot_encoded_mask):

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
    
