import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

def get_img(x, folder: str="train_images"):
    """
    Return image based on image name and folder.
    """
    data_folder = f"{path}/{folder}"
    image_path = os.path.join(data_folder, x)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def rle_decode(mask_rle: str="", shape: tuple=(1400, 2100)):
    """
    Decode rle encoded mask.

    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int)
                       for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")

def make_mask(df: pd.DataFrame, image_name: str="img.jpg",
              shape: tuple=(1400, 2100)):
    """
    Create mask based on df, image name and shape.
    """
    encoded_masks = df.loc[df["im_id"] == image_name, "EncodedPixels"]
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle_decode(label)
            masks[:, :, idx] = mask

    return masks

def make_mask_resized_dset(df: pd.DataFrame, image_name: str="img.jpg",
                           masks_dir: str="./masks",
                           shape: tuple=(320, 640)):
    """
    Create mask based on df, image name and shape.
    """
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
    df = df[df["im_id"] == image_name]
    for idx, im_name in enumerate(df["im_id"].values):
        for classidx, classid in enumerate(["Fish", "Flower", "Gravel", "Sugar"]):
            mask = cv2.imread(os.path.join(masks_dir, f"{classid}{im_name}"),
                              cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            # if mask[:,:,0].shape != (350,525):
            #     mask = cv2.resize(mask, (525,350))
            masks[:, :, classidx] = mask
    masks = masks/255
    return masks

def get_classification_label(df: pd.DataFrame, image_name: str):
    """
    Gets one-hot encoded labels. Assumes that the dataframe is coming in
    through ClassificationSteelDataset where there is a "hasMask" column.

    Returns:
        One-hot encoded torch tensor (length 4) of the label present for
        each class.
    """
    df = df[df["im_id"] == image_name]
    label = df["hasMask"].values * np.ones(4)
    return torch.from_numpy(label).float()

def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return x.transpose(2, 0, 1).astype("float32")
