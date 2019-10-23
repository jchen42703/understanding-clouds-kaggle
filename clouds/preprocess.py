import os
import cv2
from tqdm import tqdm
from pathlib import Path
from glob import glob

import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from clouds.io.utils import rle_decode, make_mask

COLAB_PATHS_DICT = {
    "train_dir": "./train_images/"
    "test_dir": "./test_images/"
    "train_out": "train640.zip"
    "test_out": "test640.zip"
    "mask_out": "masks640.zip"
}

class Preprocessor(object):
    def __init__(self, df, paths_dict=COLAB_PATHS_DICT,
                 out_shape_cv2=(640, 320), file_type=".jpg"):
        """
        Attributes:
            df: dataframe with cols ["Image_Label", "EncodedPixels"];
                the first dataframe from running `setup_train_and_sub_df(...)`
            paths_dict (dict): for all of the paths to the input and output dirs
                and files.
                Keys:
                - train_dir
                - test_dir
                - train_out: path to the output training images zip
                - test_out: path to the output test images zip
                - mask_out
                Leave as None, if not using a specific route.
            out_shape_cv2 (tuple): (w, h); reverse of numpy shaping (how
                cv2 handles its input sizing)
            file_type (str): either '.jpg' or '.png'
        """
        self.df = df
        # parsing the paths_dict dictionary
        # setting default values (in the event that the user is missing keys)
        keys_list = ["train_dir", "test_dir", "train_out", "test_out",
                     "mask_out"]
        for key in keys_list:
            setattr(self, key, None)
        # setting actual values from the dict
        for key in paths_dict.keys():
            setattr(self, key, paths_dict[key])
        # Gathering the file path lists
        if self.train_dir is not None:
            assert os.path.isdir(self.train_dir), \
                "Please make sure train_dir is a directory."
            self.train_fpaths = glob(os.path.join(self.train_dir, "*.jpg"),
                                     recursive=True)
            print(f"{len(self.train_fpaths)} training images")

        if self.test_dir is not None:
            assert os.path.isdir(self.test_dir), \
                "Please make sure test_dir is a directory."
            self.test_fpaths = glob(os.path.join(self.test_dir, "*.jpg"),
                                    recursive=True)
            print(f"{len(self.test_fpaths)} test images")

        self.out_shape_cv2 = out_shape_cv2
        self.file_type = file_type

    def execute_images(self, zip_path, img_fpaths):
        """
        Resizes input and saves the resulting images to the desired file format
        in a .zip file.
        """
        with zipfile.ZipFile(zip_path, "w") as arch:
            for fname in tqdm(img_fpaths, total=len(img_fpaths)):
                convert_images(fname, arch, self.file_type,
                               out_shape=self.out_shape_cv2)

    def execute_train_test(self):
        """
        Runs self.execute_images for the training/testing images
        """
        if self.train_out is not None:
            assert self.train_fpaths is not None, \
                "Make sure that train_dir is specified."
            self.execute_images(self.train_out, self.train_fpaths)
        if self.test_out is not None:
            assert self.test_fpaths is not None, \
                "Make sure that test_dir is specified."
            self.execute_images(self.test_out, self.test_fpaths)

    def execute_masks(self):
        """
        Creates the masks from rles in the provided dataframe and saves them
        in the desired file format inside of a .zip file.
        """
        all_img_ids = self.df["Image_Label"].apply(lambda x: x.split("_")[0]).drop_duplicates().values
        # print(f"{len(all_img_ids)}")

        with zipfile.ZipFile(self.mask_out, "w") as arch:
            for image_name in tqdm(all_img_ids):
                for label in ["Fish", "Flower", "Gravel", "Sugar"]:
                    mask = make_mask_single(self.df, label, image_name, shape=(1400, 2100))*255
                    mask = cv2.resize(mask, self.out_shape_cv2, interpolation=cv2.INTER_NEAREST)
                    output = cv2.imencode(self.file_type, mask)[1]
                    name = f"{label}{Path(image_name).stem}{self.file_type}"
                    arch.writestr(name, output)

def make_mask_single(df: pd.DataFrame, label: str, image_name: str,
                     shape: tuple=(1400, 2100)):
    """
    Create mask based on df, image name and shape.

    Args:
        df: dataframe with cols ["Image_Label", "EncodedPixels"]
    Returns:
        mask: numpy array with the user-specified shape
    """
    assert label in ["Fish", "Flower", "Gravel", "Sugar"]
    image_label = f"{image_name}_{label}"
    encoded = df.loc[df["Image_Label"] == image_label, "EncodedPixels"].values
    # handling NaNs and longer rles
    encoded = encoded[0] if len(encoded) == 1 else encoded
    mask = np.zeros((shape[0], shape[1]), dtype=np.float32)
    if encoded is not np.nan:
       mask = rle_decode(encoded)
    return mask

def convert_images(filename, arch_out, file_type, out_shape=(640, 320)):
    """
    Reads an image and converts it to a desired file format
    """
#     img = np.array(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB))
    img = np.array(cv2.imread(filename))

    img = cv2.resize(img, out_shape)
    output = cv2.imencode(file_type, img)[1]
    name = f"{Path(filename).stem}{file_type}"
    arch_out.writestr(name, output)
