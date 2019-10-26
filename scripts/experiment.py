import os
import torch
import segmentation_models_pytorch as smp
import pandas as pd

from pathlib import Path
from catalyst.dl.callbacks import AccuracyCallback, EarlyStoppingCallback, \
                                  CheckpointCallback
from catalyst.dl.runner import SupervisedRunner

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, \
                                     CosineAnnealingWarmRestarts, CyclicLR

from clouds.models import Pretrained
from clouds.io import ClassificationCloudDataset
from clouds.custom.ppv_tpr_f1 import PrecisionRecallF1ScoreCallback
from utils import get_preprocessing, get_training_augmentation, get_validation_augmentation, \
                  setup_train_and_sub_df, seed_everything

class TrainClassificationExperiment(object):
    """
    Stores the main parts of an experiment:
    - df split
    - loaders
    - model
    - optimizer
    - lr_scheduler
    - criterion
    """
    def __init__(self, args, df, id_mask_count=None):
        """
        Attributes:
            args (argparser.ArgumentParser): from `train_classification.py`
            df (pd.DataFrame): train_df from `setup_train_and_sub_df(args.dset_path)`
                The prepared training dataframe with the extra columns:
                    - im_id & label
            id_mask_count (pd.DataFrame): id_mask_count from
                `setup_train_and_sub_df(args.dset_path)`
                Different from `df` b/c only has ids that contain a mask.
                Defaults to None.
        """
        self.args = args
        split_dict = self.get_split(df, id_mask_count)
        self.loaders = self.get_loaders(df, **split_dict)
        self.model = self.get_model()
        self.opt = self.get_opt(self.model)
        self.lr_scheduler = self.get_lr_scheduler(self.opt)
        self.criterion = self.get_criterion()

    def get_split(self, train_df, id_mask_count=None):
        # setting up the train/val split with filenames
        if self.args.df_setup_type == "pos_only":
            print("Setting up df with pos only ids...")
            train_ids, valid_ids = train_test_split(id_mask_count["im_id"].values,
                                                    random_state=self.args.split_seed,
                                                    stratify=id_mask_count["count"],
                                                    test_size=self.args.test_size)
        elif self.args.df_setup_type == "regular":
            print("Setting up df normally...")
            train_ids, valid_ids = train_test_split(train_df["im_id"].drop_duplicates().values,
                                                    random_state=self.args.split_seed,
                                                    test_size=self.args.test_size)
        return {"train_ids": train_ids, "valid_ids": valid_ids}

    def get_loaders(self, train_df, train_ids, valid_ids):
        preprocessing_fn = smp.encoders.get_preprocessing_fn(self.args.model_name,
                                                             "imagenet")
        # Setting up the I/O
        train_dataset = ClassificationCloudDataset(self.args.dset_path, df=train_df,
                                                   datatype="train", im_ids=train_ids,
                                                   transforms=get_training_augmentation(self.args.aug_key),
                                                   preprocessing=get_preprocessing(preprocessing_fn))
        valid_dataset = ClassificationCloudDataset(self.args.dset_path, df=train_df,
                                                   datatype="valid", im_ids=valid_ids,
                                                   transforms=get_validation_augmentation(self.args.aug_key),
                                                   preprocessing=get_preprocessing(preprocessing_fn))

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size,
                                  shuffle=True, num_workers=self.args.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=self.args.batch_size,
                                  shuffle=False, num_workers=self.args.num_workers)

        self.train_steps = len(train_dataset) # for schedulers
        loaders = {
            "train": train_loader,
            "valid": valid_loader
        }
        return loaders

    def get_model(self):
        # setting up the classification model
        model = Pretrained(variant=self.args.model_name, num_classes=4,
                           pretrained=True, activation=None)
        return model

    def get_opt(self, model):
        assert isinstance(model, torch.nn.Module), \
            "`model` must be an instance of torch.nn.Module`"
        # fetching optimizers
        if self.args.opt.lower() == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        elif self.args.opt.lower() == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), \
                                        lr=self.args.lr, momentum=0.9, weight_decay=0.0001)
        return optimizer

    def get_lr_scheduler(self, optimizer):
        assert isinstance(optimizer, torch.optim.Optimizer), \
            "`optimizer` must be an instance of torch.optim.Optimizer"
        # fetching lr schedulers
        if self.args.scheduler.lower() == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
        elif self.args.scheduler.lower() == "cosineannealing":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.args.num_epochs)
        elif self.args.scheduler.lower() == "cosineannealingwr":
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=7, T_mult=2)
        elif self.args.scheduler.lower() == "clr":
            scheduler = CyclicLR(optimizer, base_lr=self.args.lr/10,
                                 max_lr=self.args.lr,
                                 steps_size_up=self.train_steps*2,
                                 mode="exp_range")
        print(f"LR Scheduler: {scheduler}")

        return scheduler

    def get_criterion(self):
        if self.args.loss.lower() == "bce_dice_loss":
            criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
        elif self.args.loss.lower() == "bce":
            criterion = torch.nn.BCEWithLogitsLoss()
        print(f"Criterion: {criterion}")

        return criterion

class TrainClassificationExperimentFromConfig(object):
    """
    Stores the main parts of an experiment:
    - df split
    - loaders
    - model
    - optimizer
    - lr_scheduler
    - criterion
    """
    def __init__(self, config: dict, df: pd.DataFrame, id_mask_count=None):
        """
        Attributes:
            config (dict): from `train_classification_yaml.py`
            df (pd.DataFrame): train_df from `setup_train_and_sub_df(config['dset_path'])`
                The prepared training dataframe with the extra columns:
                    - im_id & label
            id_mask_count (pd.DataFrame): id_mask_count from
                `setup_train_and_sub_df(args.dset_path)`
                Different from `df` b/c only has ids that contain a mask.
                Defaults to None.
        """
        # for reuse
        self.config = config
        self.io_params = config["io_params"]
        self.opt_params = config["opt_params"]
        self.cb_params = config["callback_params"]
        # initializing the experiment components
        split_dict = self.get_split(df, id_mask_count)
        self.loaders = self.get_loaders(df, **split_dict)
        self.model = self.get_model()
        self.opt = self.get_opt(self.model)
        self.lr_scheduler = self.get_lr_scheduler(self.opt)
        self.criterion = self.get_criterion()
        self.cb_list = self.get_callbacks(self.model)

    def get_split(self, train_df, id_mask_count=None):
        # setting up the train/val split with filenames
        df_setup_type = self.io_params["df_setup_type"].lower()
        split_seed: int = self.io_params["split_seed"]
        test_size: float = self.io_params["test_size"]

        if df_setup_type == "pos_only":
            print("Setting up df with pos only ids...")
            train_ids, valid_ids = train_test_split(id_mask_count["im_id"].values,
                                                    random_state=split_seed,
                                                    stratify=id_mask_count["count"],
                                                    test_size=test_size)
        elif df_setup_type == "regular":
            print("Setting up df normally...")
            train_ids, valid_ids = train_test_split(train_df["im_id"].drop_duplicates().values,
                                                    random_state=split_seed,
                                                    test_size=test_size)
        return {"train_ids": train_ids, "valid_ids": valid_ids}

    def get_loaders(self, train_df, train_ids, valid_ids):
        """
        Creates train/val datasets and loaders. Returns the loaders.
        """
        preprocessing_fn = smp.encoders.get_preprocessing_fn(self.config["model_name"],
                                                             "imagenet")
        preprocessing_transform = get_preprocessing(preprocessing_fn)
        train_aug = get_training_augmentation(self.io_params["aug_key"])
        val_aug = get_validation_augmentation(self.io_params["aug_key"])
        # Setting up the datasets
        train_dataset = ClassificationCloudDataset(self.config["dset_path"],
                                                   df=train_df,
                                                   datatype="train",
                                                   im_ids=train_ids,
                                                   transforms=train_aug,
                                                   preprocessing=preprocessing_transform)
        valid_dataset = ClassificationCloudDataset(self.config["dset_path"],
                                                   df=train_df,
                                                   datatype="valid",
                                                   im_ids=valid_ids,
                                                   transforms=val_aug,
                                                   preprocessing=preprocessing_transform)
        # setting up the loaders
        b_size, num_workers = self.io_params["batch_size"], self.io_params["num_workers"]
        train_loader = DataLoader(train_dataset, batch_size=b_size,
                                  shuffle=True, num_workers=num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=b_size,
                                  shuffle=False, num_workers=num_workers)

        self.train_steps = len(train_dataset) # for schedulers
        loaders = {
            "train": train_loader,
            "valid": valid_loader
        }
        return loaders

    def get_model(self):
        # setting up the classification model
        model = Pretrained(variant=self.config["model_name"], num_classes=4,
                           pretrained=True, activation=None)
        return model

    def get_opt(self, model):
        assert isinstance(model, torch.nn.Module), \
            "`model` must be an instance of torch.nn.Module`"
        # fetching optimizers
        lr = self.opt_params["lr"]
        opt_name = self.opt_params["opt"].lower()
        if opt_name == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=lr)
        elif opt_name == "sgd":
            opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), \
                                  lr=lr, momentum=0.9, weight_decay=0.0001)
        return opt

    def get_lr_scheduler(self, optimizer):
        assert isinstance(optimizer, torch.optim.Optimizer), \
            "`optimizer` must be an instance of torch.optim.Optimizer"
        sched_params = self.opt_params["scheduler_params"]
        scheduler_name = sched_params["scheduler"].lower()
        scheduler_args = sched_params[scheduler_name]
        # fetching lr schedulers
        if scheduler_name == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, **scheduler_args)
        elif scheduler_name == "cosineannealing":
            scheduler = CosineAnnealingLR(optimizer, **scheduler_args)
        elif scheduler_name == "cosineannealingwr":
            scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                    **scheduler_args)
        elif scheduler_name == "clr":
            scheduler = CyclicLR(optimizer, **scheduler_args)
        print(f"LR Scheduler: {scheduler}")

        return scheduler

    def get_criterion(self):
        loss_name = self.config["loss"].lower()
        if loss_name == "bce_dice_loss":
            criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
        elif loss_name == "bce":
            criterion = torch.nn.BCEWithLogitsLoss()
        print(f"Criterion: {criterion}")

        return criterion

    def get_callbacks(self, model=None):
        callbacks_list = [PrecisionRecallF1ScoreCallback(num_classes=4),#DiceCallback(),
                          EarlyStoppingCallback(**self.cb_params["earlystop"]),
                          AccuracyCallback(**self.cb_params["accuracy"]),
                          ]
        ckpoint_params = self.cb_params["checkpoint_params"]
        if ckpoint_params["checkpoint_path"] != None: # hacky way to say no checkpoint callback but eh what the heck
            mode = ckpoint_params["mode"].lower()
            if mode == "full":
                print("Stateful loading...")
                ckpoint_p = Path(ckpoint_params["checkpoint_path"])
                fname = ckpoint_p.name
                # everything in the path besides the base file name
                resume_dir = str(ckpoint_p.parents[0])
                print(f"Loading {fname} from {resume_dir}. \
                      Checkpoints will also be saved in {resume_dir}.")
                # adding the checkpoint callback
                callbacks_list = callbacks_list + [CheckpointCallback(resume=fname,
                                                                      resume_dir=resume_dir),]
            elif mode == "model_only":
                print("Loading weights into model...")
                model = load_weights_train(ckpoint_params["checkpoint_path"], model)
        return callbacks_list

def load_weights_train(checkpoint_path, model):
    """
    Loads weights from a checkpoint and into training.

    Args:
        checkpoint_path (str): path to a .pt or .pth checkpoint
        model (torch.nn.Module): <-
    Returns:
        Model with loaded weights and in train() mode
    """
    try:
        # catalyst weights
        state_dict = torch.load(checkpoint_path, map_location="cpu")["model_state_dict"]
    except:
        # anything else
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.train()
    return model
