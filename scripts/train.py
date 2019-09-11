import os
import torch
import pandas as pd
import segmentation_models_pytorch as smp

from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl import utils

from sklearn.model_selection import train_test_split

from cloud.io.dataset import CloudDataset
from .utils import get_preprocessing, get_training_augmentation, get_validation_augmentation

def main(path="../input/understanding_cloud_organization", num_epochs=21, bs=16, encoder="resnet50",
         test_size=0.1):
    """
    Main code for training.
    Args:
        path (str): Path to the dataset (unzipped)
        num_epochs (int): number of epochs to train for
        bs (int): batch size
        encoder (str): one of the encoders in https://github.com/qubvel/segmentation_models.pytorch
    """
    # Reading the in the .csvs
    train = pd.read_csv(f"{path}/train.csv")
    sub = pd.read_csv(f"{path}/sample_submission.csv")

    id_mask_count = train.loc[train["EncodedPixels"].isnull() == False, "Image_Label"].apply(lambda x: x.split("_")[0]).value_counts().\
    reset_index().rename(columns={"index": "img_id", "Image_Label": "count"})
    # setting up the train/val split with filenames
    train_ids, valid_ids = train_test_split(id_mask_count["img_id"].values, random_state=42,
                                            stratify=id_mask_count["count"], test_size=test_size)
    test_ids = sub["Image_Label"].apply(lambda x: x.split("_")[0]).drop_duplicates().values

    # setting up model (U-Net with ImageNet Encoders)
    ENCODER_WEIGHTS = "imagenet"
    DEVICE = "cuda"

    ACTIVATION = None
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=ENCODER_WEIGHTS,
        classes=4,
        activation=ACTIVATION,
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, ENCODER_WEIGHTS)

    # Setting up the I/O
    num_workers = 0
    train_dataset = CloudDataset(df=train, datatype="train", img_ids=train_ids,
                                 transforms=get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = CloudDataset(df=train, datatype="valid", img_ids=valid_ids,
                                 transforms=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }
    # everything is saved here (i.e. weights + stats)
    logdir = "./logs/segmentation"

    # model, criterion, optimizer
    optimizer = torch.optim.Adam([
        {"params": model.decoder.parameters(), "lr": 1e-2},
        {"params": model.encoder.parameters(), "lr": 1e-3},
    ])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
    criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
    runner = SupervisedRunner()

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=[DiceCallback(), EarlyStoppingCallback(patience=5, min_delta=0.001)],
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True
    )
    utils.plot_metrics(
        logdir=logdir,
        # specify which metrics we want to plot
        metrics=["loss", "dice", "lr", "_base/lr"]
    )


if __name__ == "__main__":
    import argparse
    # parsing the arguments from the command prompt
    parser = argparse.ArgumentParser(description="For training.")
    # parser.add_argument("--log_dir", type=str, required=True,
    #                     help="Path to the base directory where logs and weights are saved")
    parser.add_argument("--dset_path", type=str, required=True,
                        help="Path to the unzipped kaggle dataset directory.")
    parser.add_argument("--num_epochs", type=int, required=False, default=21,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, required=False, default=16,
                        help="Batch size")
    parser.add_argument("--encoder", type=str, required=False, default="resnet50",
                        help="one of the encoders in https://github.com/qubvel/segmentation_models.pytorch")
    parser.add_argument("--test_size", type=float, required=False, default=0.1,
                        help="Fraction of total dataset to make the validation set.")
    main(path=parser.dset_path, num_epochs=parser.num_epochs, bs=parser.batch_size,
         encoder=parser.encoder, test_size=parser.test_size)
