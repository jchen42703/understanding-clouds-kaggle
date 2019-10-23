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

from cloud.io.dataset import CloudDataset, ClassificationCloudDataset
from cloud.inference.inference_class import Inference
from utils import get_validation_augmentation, get_preprocessing, setup_train_and_sub_df
from parsing_utils import clean_args_create_submission_no_trace, load_classification_models

def main(args):
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
    args = clean_args_create_submission_no_trace(args)
    # setting up the test I/O
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, "imagenet")
    # setting up the train/val split with filenames
    train, sub, _ = setup_train_and_sub_df(args.dset_path)
    test_ids = sub["Image_Label"].apply(lambda x: x.split("_")[0]).drop_duplicates().values
    # datasets/data loaders
    if args.mode == "segmentation":
        test_dataset = CloudDataset(args.dset_path, df=sub, datatype="test", im_ids=test_ids,
                                    transforms=get_validation_augmentation(),
                                    preprocessing=get_preprocessing(preprocessing_fn))
        models = smp.Unet(encoder_name=args.encoder, encoder_weights=None,
                          classes=4, activation=None, attention_type=None)

    elif args.mode == "classification":
        test_dataset = ClassificationCloudDataset(args.dset_path, df=sub,
                                                  datatype="test", im_ids=test_ids,
                                                  transforms=get_validation_augmentation(),
                                                  preprocessing=get_preprocessing(preprocessing_fn))
        models = [Pretrained(variant=name, num_classes=4, pretrained=False)
                  for name in args.clf_model_names]

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    infer = Inference(args.checkpoint_paths, test_loader, test_dataset,
                      models=models, mode=args.mode, tta_flips=args.tta)
    out_df = infer.create_sub(sub=sub)

if __name__ == "__main__":
    import argparse
    # parsing the arguments from the command prompt
    parser = argparse.ArgumentParser(description="For inference.")
    parser.add_argument("--dset_path", type=str, required=True,
                        help="Path to the unzipped kaggle dataset directory.")
    parser.add_argument("--mode", type=str, required=True,
                        help="Either 'segmentation' or 'classification'")
    parser.add_argument("--clf_model_names", nargs="+", type=str, required=False,
                        default="resnet50", help="one of the models in https://github.com/Cadene/pretrained-models.pytorch")
    parser.add_argument("--batch_size", type=int, required=False, default=8,
                        help="Batch size")
    parser.add_argument("--encoder", type=str, required=False, default="resnet50",
                        help="one of the encoders in https://github.com/qubvel/segmentation_models.pytorch")
    parser.add_argument("--checkpoint_paths", nargs="+", type=str, required=False,
                        default="./logs/segmentation/checkpoints/best.pth",
                        help="Path to checkpoint that was created during training \
                        Should be corresponding to `classification_models`")
    parser.add_argument("--dropout_p", type=float, required=False, default=0.5,
                        help="Dropout probability before the final classification head.")
    parser.add_argument("--tta", nargs="+", type=str, required=False,
                        default="lr_flip",
                        help="Test time augmentation (lr_flip, ud_flip, and/or \
                        lrud_flip). Make sure to divide the flips with spaces.")
    args = parser.parse_args()
    main(args)
