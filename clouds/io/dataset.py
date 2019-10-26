import albumentations as albu
from albumentations import pytorch as AT

import pandas as pd
import numpy as np
import os
import cv2

from torch.utils.data import Dataset
from .utils import make_mask, make_mask_resized_dset, get_classification_label

class CloudDataset(Dataset):
    def __init__(self, data_folder: str, df: pd.DataFrame, im_ids: np.array,
                 masks_folder: str=None,
                 transforms=albu.Compose([albu.HorizontalFlip(), AT.ToTensor()]),
                 preprocessing=None):
        """
        Attributes
            data_folder (str): path to the image directory
            df (pd.DataFrame): dataframe with the labels
            im_ids (np.ndarray): of image names.
            masks_folder (str): path to the masks directory
                assumes `use_resized_dataset == True`
            transforms (albumentations.augmentation): transforms to apply
                before preprocessing. Defaults to HFlip and ToTensor
            preprocessing: ops to perform after transforms, such as
                z-score standardization. Defaults to None.
        """
        self.df = df
        self.data_folder = data_folder
        self.masks_folder = masks_folder
        if isinstance(masks_folder, str):
            self.use_resized_dataset = True
            print(f"Using resized masks in {masks_folder}...")
        self.img_ids = im_ids
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        if not self.use_resized_dataset:
            mask = make_mask(self.df, image_name)
        else:
            mask = make_mask_resized_dset(self.df, image_name,
                                          self.masks_folder)
        # loading image
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # apply augmentations
        augmented = self.transforms(image=img, mask=mask)
        img = augmented["image"]
        mask = augmented["mask"]
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed["image"]
            mask = preprocessed["mask"]
        return img, mask

    def __len__(self):
        return len(self.img_ids)

class ClassificationCloudDataset(Dataset):
    def __init__(self, data_folder: str, df: pd.DataFrame, im_ids: np.array,
                 transforms=albu.Compose([albu.HorizontalFlip(), AT.ToTensor()]),
                 preprocessing=None):
        """
        Attributes
            data_folder (str): path to the image directory
            df (pd.DataFrame): dataframe with the labels
            im_ids (np.ndarray): of image names.
            transforms (albumentations.augmentation): transforms to apply
                before preprocessing. Defaults to HFlip and ToTensor
            preprocessing: ops to perform after transforms, such as
                z-score standardization. Defaults to None.
        """
        df["hasMask"] = ~ df["EncodedPixels"].isna()
        self.df = df
        self.data_folder = data_folder
        self.img_ids = im_ids
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        # loading image
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        class_label = get_classification_label(self.df, image_name)
        # apply augmentations
        augmented = self.transforms(image=img)
        img = augmented["image"]
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=None)
            img = preprocessed["image"]
        return img, class_label

    def __len__(self):
        return len(self.img_ids)
