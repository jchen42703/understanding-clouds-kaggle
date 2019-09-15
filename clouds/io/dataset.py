import albumentations as albu
from albumentations import torch as AT

import pandas as pd
import numpy as np
import os
import cv2

from torch.utils.data import Dataset
from .utils import make_mask, make_mask_resized_dset

class CloudDataset(Dataset):
    def __init__(self, path: str, df: pd.DataFrame=None, datatype: str="train", im_ids: np.array=None,
                 transforms=albu.Compose([albu.HorizontalFlip(), AT.ToTensor()]),
                 preprocessing=None, use_resized_dataset=False):
        self.df = df
        if datatype != 'test':
            self.data_folder = f"{path}/train_images"
        else:
            self.data_folder = f"{path}/test_images"
        self.masks_folder = os.path.join(path, "masks") # only when use_resized_dataset=True
        self.use_resized_dataset = use_resized_dataset
        self.img_ids = im_ids
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        if not self.use_resized_dataset:
            mask = make_mask(self.df, image_name)
        else:
            mask = make_mask_resized_dset(self.df, image_name, self.masks_folder)
        # loading image
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # apply augmentations
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
        return img, mask

    def __len__(self):
        return len(self.img_ids)
