import os
import torch
import pandas as pd
import segmentation_models_pytorch as smp

from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl import utils

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from pathlib import Path

from clouds.io import CloudDataset
from utils import get_preprocessing, get_training_augmentation, get_validation_augmentation, setup_train_and_sub_df, seed_everything

def main(args):
    """
    Main code for training for training a U-Net with some user-defined encoder.
    Args:
        args (instance of argparse.ArgumentParser): arguments must be compiled with parse_args
    Returns:
        None
    """
    # setting up the train/val split with filenames
    train, sub, id_mask_count = setup_train_and_sub_df(args.dset_path)
    # setting up the train/val split with filenames
    seed_everything(args.split_seed)
    train_ids, valid_ids = train_test_split(id_mask_count["im_id"].values, random_state=args.split_seed,
                                            stratify=id_mask_count["count"], test_size=args.test_size)
    # setting up model (U-Net with ImageNet Encoders)
    ENCODER_WEIGHTS = "imagenet"
    DEVICE = "cuda"

    attention_type = None if args.attention_type == "None" else args.attention_type
    model = smp.Unet(encoder_name=args.encoder, encoder_weights=ENCODER_WEIGHTS,
                     classes=4, activation=None, attention_type=attention_type
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, ENCODER_WEIGHTS)

    # Setting up the I/O
    train_dataset = CloudDataset(args.dset_path, df=train, datatype="train", im_ids=train_ids,
                                 transforms=get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn),
                                 use_resized_dataset=args.use_resized_dataset)
    valid_dataset = CloudDataset(args.dset_path, df=train, datatype="valid", im_ids=valid_ids,
                                 transforms=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn),
                                 use_resized_dataset=args.use_resized_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }
    # everything is saved here (i.e. weights + stats)
    logdir = "./logs/segmentation"

    # model, criterion, optimizer
    optimizer = torch.optim.Adam([
        {"params": model.decoder.parameters(), "lr": args.encoder_lr},
        {"params": model.encoder.parameters(), "lr": args.decoder_lr},
    ])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
    criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
    runner = SupervisedRunner()

    callbacks_list = [DiceCallback(), EarlyStoppingCallback(patience=5, min_delta=0.001),]
    if args.checkpoint_path != "None": # hacky way to say no checkpoint callback but eh what the heck
        ckpoint_p = Path(args.checkpoint_path)
        fname = ckpoint_p.name
        resume_dir = str(ckpoint_p.parents[0]) # everything in the path besides the base file name
        print(f"Loading {fname} from {resume_dir}. Checkpoints will also be saved in {resume_dir}.")
        callbacks_list = callbacks_list + [CheckpointCallback(resume=fname, resume_dir=resume_dir),]

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=callbacks_list,
        logdir=logdir,
        num_epochs=args.num_epochs,
        verbose=True
    )

if __name__ == "__main__":
    import argparse
    from parsing_utils import add_bool_arg

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
    parser.add_argument("--encoder_lr", type=float, required=False, default=0.00001,
                        help="Learning rate for the encoder.")
    parser.add_argument("--decoder_lr", type=float, required=False, default=0.001,
                        help="Learning rate for the decoder.")
    parser.add_argument("--encoder", type=str, required=False, default="resnet50",
                        help="one of the encoders in https://github.com/qubvel/segmentation_models.pytorch")
    parser.add_argument("--test_size", type=float, required=False, default=0.1,
                        help="Fraction of total dataset to make the validation set.")
    add_bool_arg(parser, "use_resized_dataset", default=False)
    parser.add_argument("--split_seed", type=int, required=False, default=42,
                        help="Seed for the train/val dataset split")
    parser.add_argument("--num_workers", type=int, required=False, default=2,
                        help="Number of workers for data loaders.")
    parser.add_argument("--attention_type", type=str, required=False, default="scse",
                        help="Attention type; if you want None, just put the string None.")
    parser.add_argument("--checkpoint_path", type=str, required=False, default="None",
                        help="Checkpoint path; if you want to train from scratch, just put the string as None.")
    args = parser.parse_args()

    main(args)
