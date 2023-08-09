import albumentations
from albumentations import pytorch as AT
from PIL import Image
from torchvision import transforms
import random
import numpy as np


class AddPepperNoise(object):

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):

        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct / 2., noise_pct / 2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255
            img_[mask == 2] = 0
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img


# albumentations transforms
def at_transform(args):
    train_transforms = albumentations.Compose([
        albumentations.Resize(args["resize_h"], args["resize_w"]),
        albumentations.Sharpen(p=0.5),
        albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=7),
            albumentations.MedianBlur(blur_limit=7),
            albumentations.GaussianBlur(blur_limit=(3, 7)),
            albumentations.GaussNoise(var_limit=(10, 50)),
        ], p=0.8 if args["Blur"] == 1 else 0),
        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=2.0),  # 光学畸变
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),  # 弹性变换
        ], p=0.5 if args["OGE"] == 1 else 0),
        albumentations.CLAHE(clip_limit=4, p=0.5 if args["CLAHE"] == 1 else 0),
        albumentations.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=0, p=0.5),
        albumentations.CoarseDropout(max_holes=1, max_height=int(args["resize_h"] * 0.3),
                                     max_width=int(args["resize_w"] * 0.3), min_height=1, min_width=1, p=0.5 if args["Cutout"] == 1 else 0),
        albumentations.Normalize(0.21191326, 0.21312135),
        AT.ToTensorV2()
    ])

    val_transforms = albumentations.Compose([
        albumentations.Resize(args["resize_h"], args["resize_w"]),
        albumentations.Normalize(0.21191326, 0.21312135),
        AT.ToTensorV2()
    ])

    test_transforms = albumentations.Compose([
        albumentations.Resize(args["resize_h"], args["resize_w"]),
        albumentations.Normalize(0.21191326, 0.21312135),
        AT.ToTensorV2()
    ])

    return train_transforms, val_transforms, test_transforms


# torchvision transforms
def tv_transform(args):
    train_transforms = transforms.Compose([
        transforms.Resize((args["resize_h"], args["resize_w"])),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0),
        transforms.RandomRotation(5),
        AddPepperNoise(0.95, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(0.21191326, 0.21312135)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((args["resize_h"], args["resize_w"])),
        transforms.ToTensor(),
        transforms.Normalize(0.21191326, 0.21312135)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((args["resize_h"], args["resize_w"])),
        transforms.ToTensor(),
        transforms.Normalize(0.21191326, 0.21312135)
    ])

    return train_transforms, val_transforms, test_transforms
