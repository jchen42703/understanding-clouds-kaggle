import gc
import os
import tqdm
import cv2
import torch
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import pickle

from torch.utils.data import DataLoader

from clouds.models import Pretrained
from clouds.io import CloudDataset, ClassificationCloudDataset
from clouds.inference import PseudoLabeler
from clouds.experiments.utils import get_validation_augmentation, \
                                     get_preprocessing, setup_train_and_sub_df

def main(config):
    """
    Main code for creating the segmentation-only submission file. All masks are
    converted to either "" or RLEs

    Args:
        args (instance of argparse.ArgumentParser): arguments must be compiled with parse_args
    Returns:
        None
    """
    torch.cuda.empty_cache()
    gc.collect()
    # setting up the test I/O
    # setting up the train/val split with filenames
    train_csv_path = config["train_csv_path"]
    sample_sub_csv_path = config["sample_sub_csv_path"]
    train_df, sub, _ = setup_train_and_sub_df(train_csv_path, sample_sub_csv_path)
    test_ids = sub["Image_Label"].apply(lambda x: x.split("_")[0]).drop_duplicates().values
    print(f"# of test ids: {len(test_ids)}")
    n_encoded = len(sub["EncodedPixels"])
    print(f"length of sub: {n_encoded}")
    # datasets/data loaders
    io_params = config["io_params"]
    preprocessing_fn = smp.encoders.get_preprocessing_fn(config["model_names"][0],
                                                         "imagenet")
    preprocessing_transform = get_preprocessing(preprocessing_fn)
    val_aug = get_validation_augmentation(io_params["aug_key"])
    # fetching the proper datasets and models
    print("Assuming that all models are from the same family...")
    if config["mode"] == "segmentation":
        test_dataset = CloudDataset(io_params["image_folder"], df=sub,
                                    im_ids=test_ids,
                                    transforms=val_aug,
                                    preprocessing=preprocessing_transform)
        models = [smp.Unet(encoder_name=name, encoder_weights=None,
                           classes=4, activation=None, attention_type=None)
                  for name in config["model_names"]]

    elif config["mode"] == "classification":
        test_dataset = ClassificationCloudDataset(io_params["image_folder"],
                                                  df=sub, im_ids=test_ids,
                                                  transforms=val_aug,
                                                  preprocessing=preprocessing_transform)
        models = [Pretrained(variant=name, num_classes=4, pretrained=False)
                  for name in config["model_names"]]

    test_loader = DataLoader(test_dataset, batch_size=io_params["batch_size"],
                             shuffle=False, num_workers=io_params["num_workers"])
    pseudo = PseudoLabeler(config["checkpoint_paths"], test_loader,
                           models=models, mode=config["mode"],
                           **config["pseudo_params"])
    pseudo.create_clf_pseudo_df(sub=sub, **config["hard_labels_params"])

if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser(description="For training.")
    parser.add_argument("--yml_path", type=str, required=True,
                        help="Path to the .yml config.")
    args = parser.parse_args()

    with open(args.yml_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    main(config)
