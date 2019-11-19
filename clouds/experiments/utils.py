import albumentations as albu
import pandas as pd
import os
import random
import numpy as np
import torch

from clouds.io.utils import to_tensor

def setup_train_and_sub_df(train_csv_path, sample_sub_csv_path):
    """
    Sets up the training and sample submission DataFrame.
    Args:
        train_csv_path (str): path to 'train.csv'
        sample_sub_csv_path (str): path to `sample_submission.csv`
    Returns:
        tuple of:
            train (pd.DataFrame): The prepared training dataframe with the extra columns:
                im_id & label
            sub (pd.DataFrame): The prepared sample submission dataframe with the
                same extra columns as train
            id_mask_count (pd.DataFrame): The dataframe prepared for splitting
    """
    # Reading the in the .csvs
    train = pd.read_csv(train_csv_path)
    sub = pd.read_csv(sample_sub_csv_path)

    # setting the dataframe for training/inference
    train["label"] = train["Image_Label"].apply(lambda x: x.split("_")[1])
    train["im_id"] = train["Image_Label"].apply(lambda x: x.split("_")[0])

    sub["label"] = sub["Image_Label"].apply(lambda x: x.split("_")[1])
    sub["im_id"] = sub["Image_Label"].apply(lambda x: x.split("_")[0])
    id_mask_count = train.loc[train["EncodedPixels"].isnull() == False, "Image_Label"].apply(lambda x: x.split("_")[0]).value_counts().\
    reset_index().rename(columns={"index": "im_id", "Image_Label": "count"})
    return (train, sub, id_mask_count)

def get_training_augmentation(augmentation_key="aug5"):
    transform_dict = {
                      "aug1": [
                                albu.HorizontalFlip(p=0.5),
                                albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0,
                                                      shift_limit=0.1, p=0.5,
                                                      border_mode=0),
                                albu.GridDistortion(p=0.5),
                                albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
	                          ],
                      "aug2": [
                                albu.HorizontalFlip(p=0.5),
                                albu.VerticalFlip(p=0.5),
                                albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0,
                                                      shift_limit=0.5, p=0.5,
                                                      border_mode=0),
                                albu.OneOf([
                                            albu.RandomResizedCrop(height=320, width=640,
                                                                   scale=(1.0, 0.9),
                                                                   ratio=(0.75, 1.33)),
                                            albu.RandomCrop(height=320, width=640)
                                           ], p=0.3),
                                albu.Lambda(image=do_random_log_contrast, p=0.5),
                                albu.Lambda(image=do_noise, p=0.5),
                              ],
                      "aug3": [
                                albu.HorizontalFlip(p=0.5),
                                albu.VerticalFlip(p=0.5),
                                albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0,
                                                      shift_limit=0.5, p=0.5,
                                                      border_mode=0),
                                albu.RandomCrop(height=320, width=640, p=1),
                                albu.Lambda(image=do_random_log_contrast, p=0.5),
                                albu.Lambda(image=do_noise, p=0.5),
                              ],
                      "aug4": [
                                albu.HorizontalFlip(p=0.5),
                                albu.VerticalFlip(p=0.5),
                                albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0,
                                                      shift_limit=0.5, p=0.5,
                                                      border_mode=0),
                                albu.RandomCrop(height=320, width=320, p=1),
                                albu.Lambda(image=do_noise, p=0.5),
                              ],
                      "aug5": [
                                albu.HorizontalFlip(p=0.5),
                                albu.VerticalFlip(p=0.5),
                                albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0,
                                                      shift_limit=0.5, p=0.5,
                                                      border_mode=0),
                                albu.RandomCrop(height=320, width=320, p=1),
                                albu.Lambda(image=do_noise, p=0.5),
                                albu.GridDistortion(p=0.5),
                                albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
                              ],
                      "aug6": [
                                albu.HorizontalFlip(p=0.5),
                                albu.VerticalFlip(p=0.5),
                                albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0,
                                                      shift_limit=0.5, p=0.5,
                                                      border_mode=0),
                                albu.OneOf([
                                            albu.RandomResizedCrop(height=696, width=1048,
                                                                   scale=(1.0, 0.9),
                                                                   ratio=(0.75, 1.33)),
                                            albu.RandomCrop(height=696, width=1048)
                                           ], p=1),
                                albu.Lambda(image=do_noise, p=0.5),
                              ],
                      "aug7": [
                                albu.HorizontalFlip(p=0.5),
                                albu.VerticalFlip(p=0.5),
                                albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0,
                                                      shift_limit=0.5, p=0.5,
                                                      border_mode=0),
                                albu.OneOf([
                                            albu.RandomResizedCrop(height=688, width=1040,
                                                                   scale=(1.0, 0.9),
                                                                   ratio=(0.75, 1.33)),
                                            albu.RandomCrop(height=688, width=1040)
                                           ], p=1),
                                albu.Lambda(image=do_noise, p=0.5),
                              ],
                      "aug8": [
                                albu.HorizontalFlip(p=0.5),
                                albu.VerticalFlip(p=0.5),
                                albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0,
                                                      shift_limit=0.5, p=0.5,
                                                      border_mode=0),
                                albu.OneOf([
                                            albu.RandomResizedCrop(height=696, width=1048,
                                                                   scale=(1.0, 0.9),
                                                                   ratio=(0.75, 1.33)),
                                            albu.RandomCrop(height=696, width=1048)
                                           ], p=0.3),
                                ],
                     }
    train_transform = transform_dict[augmentation_key]
    return albu.Compose(train_transform)

def get_validation_augmentation(augmentation_key="aug5"):
    """
    Validation transforms
    """
    transform_dict = {
                      "aug1": [],
                      "aug2": [],
                      "aug3": [albu.RandomCrop(height=320, width=640, p=1)],
                      "aug4": [albu.RandomCrop(height=320, width=320, p=1)],
                      "aug5": [albu.RandomCrop(height=320, width=320, p=1)],
                      "aug6": [albu.RandomCrop(height=696, width=1048, p=1)],
                      "aug7": [albu.RandomCrop(height=688, width=1040, p=1)],
                      "aug8": [],
                     }
    test_transform = transform_dict[augmentation_key]
    return albu.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def do_random_log_contrast(image, **kwargs):
    gain = np.random.uniform(0.70,1.30,1)
    inverse = np.random.choice(2,1)

    image = image.astype(np.float32)/255
    if inverse==0:
        image = gain*np.log(image+1)
    else:
        image = gain*(2**image-1)

    image = np.clip(image*255, 0, 255).astype(np.uint8)
    return image

def do_noise(image, noise=8, **kwargs):
    H,W = image.shape[:2]
    image = image + np.random.uniform(-1, 1, (H, W, 1))*noise
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
