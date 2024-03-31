import warnings
import albumentations as A
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
    
# --------------------------------------------------------------------
# VOC Dataset
# --------------------------------------------------------------------

VOC_post_transform = A.Lambda(name="post_transform", image=post_transform)

VOC_train_transform = A.Compose(
    [
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.RandomCrop(256, 256),
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.05, rotate_limit=15, p=0.4),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        VOC_post_transform,
    ]
)

VOC_val_transform = A.Compose([
    VOC_post_transform,
])

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
             A.IAASharpen(alpha=(0.1, 0.22), p=0.3),
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

# crop 768 and hard augs
train_transform_2 = A.Compose([

    A.RandomScale(scale_limit=0.3, p=0.5),
    A.PadIfNeeded(768, 768, p=1),
    A.RandomCrop(768, 768, p=1.),
    A.Flip(p=0.75),
    A.Downscale(scale_min=0.5, scale_max=0.75, p=0.05),

    # color transforms
    A.OneOf(
        [
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
            A.ChannelShuffle(p=0.2),
            A.HueSaturationValue(p=1),
            A.RGBShift(p=1),
        ],
        p=0.5,
    ),

    # noise transforms
    A.OneOf(
        [
            A.GaussNoise(p=1),
            A.MultiplicativeNoise(p=1),
            A.IAASharpen(p=1),
            # A.ImageCompression(quality_lower=0.7, p=1),
            A.GaussianBlur(p=1),
        ],
        p=0.2,
    ),
    post_transform,
])

valid_transform_2 = valid_transform_1
test_transform_2 = test_transform_1

# crop 1024 and hard augs
train_transform_3 = A.Compose([

    A.RandomScale(scale_limit=0.3, p=0.5),
    A.PadIfNeeded(1024, 1024, p=1),
    A.RandomCrop(1024, 1024, p=1.),
    A.Flip(p=0.75),
    A.Downscale(scale_min=0.5, scale_max=0.75, p=0.05),

    # color transforms
    A.OneOf(
        [
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
            A.ChannelShuffle(p=0.2),
            A.HueSaturationValue(p=1),
            A.RGBShift(p=1),
        ],
        p=0.5,
    ),

    # noise transforms
    A.OneOf(
        [
            A.GaussNoise(p=1),
            A.MultiplicativeNoise(p=1),
            A.IAASharpen(p=1),
            # A.ImageCompression(quality_lower=0.7, p=1),
            A.GaussianBlur(p=1),
        ],
        p=0.2,
    ),
    post_transform,
])

valid_transform_3 = valid_transform_1
test_transform_3 = test_transform_1

# crop 768 and very hard augs
train_transform_4 = A.Compose([

    A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=45, border_mode=0, value=0, p=0.7),
    A.PadIfNeeded(768, 768, border_mode=0, value=0, p=1.),
    A.RandomCrop(768, 768, p=1.),
    A.Flip(p=0.75),
    A.Downscale(scale_min=0.5, scale_max=0.75, p=0.05),
    A.MaskDropout(max_objects=3, image_fill_value=0, mask_fill_value=0, p=0.1),

    # color transforms
    A.OneOf(
        [
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
            A.RandomGamma(gamma_limit=(70, 130), p=1),
            A.ChannelShuffle(p=0.2),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=1),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1),
        ],
        p=0.8,
    ),

    # distortion
    A.OneOf(
        [
            A.ElasticTransform(p=1),
            A.OpticalDistortion(p=1),
            A.GridDistortion(p=1),
            A.IAAPerspective(p=1),
        ],
        p=0.2,
    ),

    # noise transforms
    A.OneOf(
        [
            A.GaussNoise(p=1),
            A.MultiplicativeNoise(p=1),
            A.IAASharpen(p=1),
            A.GaussianBlur(p=1),
        ],
        p=0.2,
    ),
    post_transform,
])

valid_transform_4 = valid_transform_1
test_transform_4 = test_transform_1