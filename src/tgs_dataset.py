# coding: utf-8

# #### Reimplement torch.utils.data.Dataset for TGS.
# To-do list:
# 1) Figure out functioning of grayscale_aug() and hsv_aug()
# 2) Consider including more augmentation options:
# https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9

import cv2
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Compose

def load_image(path):
    img = cv2.imread(path)
    img = cv2.copyMakeBorder(img, 0, 0, 1, 1, cv2.BORDER_REPLICATE) # pad borders using existing col/row
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert img from one color space to another
    return img.astype(np.uint8)

def load_mask(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = np.asarray(img) > 255 * 0.5
            img = cv2.copyMakeBorder(img.astype(np.uint8), 0,0,1,1,cv2.BORDER_REPLICATE)
            return img.astype(np.float32)

img_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class TGS_Dataset(torch.utils.data.Dataset):

    def __init__(self, root : Path, to_augment=False):
        train_path = 'train/images'
        mask_path = 'train/masks'
        test_path = 'test/images'

        self.train_ids= [(str(root/train_path)+'/'+str(id)) for id in next(os.walk(root/train_path))[2]]
        self.mask_ids = [(str(root/mask_path)+'/'+str(id)) for id in next(os.walk(root/mask_path))[2]]
        self.test_ids = [(str(root/test_path)+'/'+str(id)) for id in next(os.walk(root/test_path))[2]]
        self.to_augment = to_augment

    def __getitem__(self, index):
        img = load_image(self.train_ids[index])
        mask = load_mask(self.mask_ids[index])
        if self.to_augment:
            img, mask = augment(img, mask)
        return img_transform(img), torch.from_numpy(np.expand_dims(mask,0))

    def __len__(self):
        return len(self.train_ids)


def grayscale_aug(img, mask):
    salt_pixels = (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)*img).astype(np.uint8) # convert mask to RGB
    gray_salt = cv2.cvtColor(salt_pixels, cv2.COLOR_RGB2GRAY) # convert RGB salt to gray
    rgb_gray_salt = cv2.cvtColor(gray_salt, cv2.COLOR_GRAY2RGB) # convert gray to RGB
    rgb_img = img.copy()
    rgb_img[rgb_gray_salt>0] = rgb_gray_salt[rgb_gray_salt>0]
    return rgb_img

def hsv_aug(img,
            hue_shift_limit = (-180,180),
            sat_shift_limit = (-255,255),
            val_shift_limit = (-255,255)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(img)
    hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
    h = cv2.add(h, hue_shift)
    sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
    s = cv2.add(s, hue_shift)
    hue_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
    v = cv2.add(v, hue_shift)
    img = cv2.merge((h,s,v))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img

def augment(img, mask):
    if np.random.random()<0.5:
        img = np.flip(img, axis=1)
        mask = np.flip(mask, axis=1)

    if np.random.random()<0.5:
        if np.random.random()<0.5:
            img = hsv_aug(img, hue_shift_limit=(-50,50), sat_shift_limit=(-5,5), val_shift_limit=(-15,15))
        else:
            img = grayscale_aug(img, mask)
    return img.copy(), mask.copy()

