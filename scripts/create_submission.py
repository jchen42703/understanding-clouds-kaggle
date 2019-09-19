import tqdm
import gc
import os
import segmentation_models_pytorch as smp
import tqdm
import cv2
import torch
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader
from catalyst.dl.callbacks import InferCallback, CheckpointCallback
from catalyst.dl.runner import SupervisedRunner

from clouds.io.dataset import CloudDataset
from clouds.io.utils import post_process, mask2rle, sigmoid
from utils import get_validation_augmentation, get_preprocessing, setup_train_and_sub_df
from clouds.inference.inference import get_encoded_pixels

def main(path, bs=8, encoder="resnet34", attention_type="scse"):
    """
    Args:
        path (str): Path to the dataset (unzipped)
        bs (int): batch size
        encoder (str): one of the encoders in https://github.com/qubvel/segmentation_models.pytorch
    """
    torch.cuda.empty_cache()
    gc.collect()

    ENCODER_WEIGHTS = "imagenet"
    ACTIVATION = None
    attention_type = None if attention_type == "None" else attention_type
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=ENCODER_WEIGHTS,
        classes=4,
        activation=ACTIVATION,
        attention_type=attention_type
    )
    # setting up the test I/O
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, ENCODER_WEIGHTS)
    # setting up the train/val split with filenames
    train, sub, _ = setup_train_and_sub_df(path)
    # train_ids, valid_ids = train_test_split(id_mask_count["im_id"].values, random_state=42,
    #                                         stratify=id_mask_count["count"], test_size=test_size)
    test_ids = sub["Image_Label"].apply(lambda x: x.split("_")[0]).drop_duplicates().values
    # datasets/data loaders
    # valid_dataset = CloudDataset(path, df=train, datatype="valid", im_ids=valid_ids,
                                 # transforms=get_validation_augmentation(),
                                 # preprocessing=get_preprocessing(preprocessing_fn))
    test_dataset = CloudDataset(path, df=sub, datatype="test", im_ids=test_ids,
                                transforms=get_validation_augmentation(),
                                preprocessing=get_preprocessing(preprocessing_fn))
    # valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=0)

    runner = SupervisedRunner()
    # loaders = {"valid": valid_loader, "test": test_loader}
    loaders = {"test": test_loader}

    create_submission(model=model, loaders=loaders, runner=runner, sub=sub)

def create_submission(model, loaders, runner, sub, class_params="default"):
    """
    runner: with .infer set
    Args:
        model (nn.Module): Segmentation module that outputs logits
        loaders: dictionary of data loaders with at least the key: "test"
        runner (an instance of a catalyst.dl.runner.SupervisedRunner):
        sub (pandas.DataFrame): sample submission dataframe. This is used to
            create the final submission dataframe.
        class_params (dict): with keys class: (threshold, minimum component size)
    """
    if class_params == "default":
        class_params = {0: (0.75, 10000), 1: (0.5, 10000), 2: (0.7, 10000), 3: (0.45, 10000)}
    assert isinstance(class_params, dict)

    logdir = "./logs/segmentation"
    ckpoint_path = os.path.join(logdir, "checkpoints", "best.pth")
    runner.infer(model=model, loaders=loaders, callbacks=[
            CheckpointCallback(
                resume=ckpoint_path,)
        ], verbose=True
    )
    print("Converting predicted masks to rle's...")
    encoded_pixels = get_encoded_pixels(loaders=loaders, runner=runner,
                                        class_params=class_params)
    # Saving the submission dataframe
    sub["EncodedPixels"] = encoded_pixels
    save_path = os.path.join(os.getcwd(), "submission.csv")
    sub.to_csv(save_path, columns=["Image_Label", "EncodedPixels"], index=False)
    print(f"Saved the submission file at {save_path}")

if __name__ == "__main__":
    import argparse
    # parsing the arguments from the command prompt
    parser = argparse.ArgumentParser(description="For inference.")
    # parser.add_argument("--log_dir", type=str, required=True,
    #                     help="Path to the base directory where logs and weights are saved")
    parser.add_argument("--dset_path", type=str, required=True,
                        help="Path to the unzipped kaggle dataset directory.")
    parser.add_argument("--batch_size", type=int, required=False, default=8,
                        help="Batch size")
    parser.add_argument("--encoder", type=str, required=False, default="resnet50",
                        help="one of the encoders in https://github.com/qubvel/segmentation_models.pytorch")
    parser.add_argument("--attention_type", type=str, required=False, default="scse",
                        help="Attention type; if you want None, just put the string None.")
    args = parser.parse_args()
    main(args.dset_path, bs=args.batch_size, encoder=args.encoder, attention_type=args.attention_type)
