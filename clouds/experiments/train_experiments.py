import os
import torch
import segmentation_models_pytorch as smp
import pandas as pd

from abc import abstractmethod
from pathlib import Path
from catalyst.dl.callbacks import AccuracyCallback, EarlyStoppingCallback, \
                                  CheckpointCallback, PrecisionRecallF1ScoreCallback
from catalyst.dl.runner import SupervisedRunner

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, \
                                     CosineAnnealingWarmRestarts, CyclicLR
from clouds.models import Pretrained, ResNet34FPN
from clouds.metrics import BCEDiceLoss, FocalLoss, HengFocalLoss
from clouds.io import ClassificationCloudDataset, CloudDataset, \
                      ClfSegCloudDataset
from .utils import get_preprocessing, get_training_augmentation, \
                   get_validation_augmentation, setup_train_and_sub_df, \
                   seed_everything

class TrainExperiment(object):
    def __init__(self, config: dict):
        """
        Args:
            config (dict): from `train_classification_yaml.py`

        Attributes:
            config-related:
                config (dict): from `train_classification_yaml.py`
                io_params (dict): contains io-related parameters
                    image_folder (key: str): path to the image folder
                    df_setup_type (key: str): regular or pos_only
                    test_size (key: float): split size for test
                    split_seed (key: int): seed
                    batch_size (key: int): <-
                    num_workers (key: int): # of workers for data loaders
                    aug_key (key: str): One of the augmentation keys for
                        `get_training_augmentation` and `get_validation_augmentation`
                        in `scripts/utils.py`
                opt_params (dict): optimizer related parameters
                    lr (key: str): learning rate
                    opt (key: str): optimizer name
                        Currently, only supports sgd and adam.
                    scheduler_params (key: str): dict of:
                        scheduler (key: str): scheduler name
                        {scheduler} (key: dict): args for the above scheduler
                cb_params (dict):
                    earlystop (key: str):
                        dict -> kwargs for EarlyStoppingCallback
                    accuracy (key: str):
                        dict -> kwargs for AccuracyCallback
                    checkpoint_params (key: dict):
                      checkpoint_path (key: str): path to the checkpoint
                      checkpoint_mode (key: str): model_only or
                        full (for stateful loading)
            split_dict (dict): train_ids and valid_ids
            train_dset, val_dset: <-
            loaders (dict): train/validation loaders
            model (torch.nn.Module): <-
            opt (torch.optim.Optimizer): <-
            lr_scheduler (torch.optim.lr_scheduler): <-
            criterion (torch.nn.Module): <-
            cb_list (list): list of catalyst callbacks
        """
        # for reuse
        self.config = config
        self.io_params = config["io_params"]
        self.opt_params = config["opt_params"]
        self.cb_params = config["callback_params"]
        self.model_params = config["model_params"]
        self.criterion_params = config["criterion_params"]
        # initializing the experiment components
        self.df, _, self.id_mask_count = self.setup_df()
        train_ids, val_ids = self.get_split()
        self.train_dset, self.val_dset = self.get_datasets(train_ids, val_ids)
        self.loaders = self.get_loaders()
        self.model = self.get_model()
        self.opt = self.get_opt()
        self.lr_scheduler = self.get_lr_scheduler()
        self.criterion = self.get_criterion()
        self.cb_list = self.get_callbacks()

    @abstractmethod
    def get_datasets(self, train_ids, valid_ids):
        """
        Initializes the data augmentation and preprocessing transforms. Creates
        and returns the train and validation datasets.
        """
        return

    @abstractmethod
    def get_model(self):
        """
        Creates and returns the model.
        """
        return

    def setup_df(self):
        """
        Setting up the dataframe to have the `im_id` & `label` columns;
            im_id: the base img name
            label: the label name
        """
        train_csv_path = self.config["train_csv_path"]
        sample_sub_csv_path = self.config["sample_sub_csv_path"]
        return setup_train_and_sub_df(train_csv_path, sample_sub_csv_path)

    def get_split(self):
        """
        Creates train/valid filename splits
        """
        # setting up the train/val split with filenames
        df_setup_type = self.io_params["df_setup_type"].lower()
        split_seed: int = self.io_params["split_seed"]
        test_size: float = self.io_params["test_size"]
        # doing the splits
        if df_setup_type == "pos_only":
            print("Splitting the df with pos only ids...")
            assert self.id_mask_count is not None
            train_ids, valid_ids = train_test_split(self.id_mask_count["im_id"].values,
                                                    random_state=split_seed,
                                                    stratify=self.id_mask_count["count"],
                                                    test_size=test_size)
        elif df_setup_type == "regular":
            print("Splitting the df normally...")
            train_ids, valid_ids = train_test_split(self.df["im_id"].drop_duplicates().values,
                                                    random_state=split_seed,
                                                    test_size=test_size)
        return (train_ids, valid_ids)

    def get_loaders(self):
        """
        Creates train/val loaders from datasets created in self.get_datasets.
        Returns the loaders.
        """
        # setting up the loaders
        b_size, num_workers = self.io_params["batch_size"], self.io_params["num_workers"]
        train_loader = DataLoader(self.train_dset, batch_size=b_size,
                                  shuffle=True, num_workers=num_workers)
        valid_loader = DataLoader(self.val_dset, batch_size=b_size,
                                  shuffle=False, num_workers=num_workers)

        self.train_steps = len(self.train_dset) # for schedulers
        return {"train": train_loader, "valid": valid_loader}

    def get_opt(self):
        assert isinstance(self.model, torch.nn.Module), \
            "`model` must be an instance of torch.nn.Module`"
        # fetching optimizers
        lr = self.opt_params["lr"]
        opt_name = self.opt_params["opt"].lower()
        if opt_name == "adam":
            opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif opt_name == "sgd":
            opt = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                  self.model.parameters()),
                                  lr=lr, momentum=0.9, weight_decay=0.0001)
        return opt

    def get_lr_scheduler(self):
        assert isinstance(self.opt, torch.optim.Optimizer), \
            "`optimizer` must be an instance of torch.optim.Optimizer"
        sched_params = self.opt_params["scheduler_params"]
        scheduler_name = sched_params["scheduler"].lower()
        scheduler_args = sched_params[scheduler_name]
        # fetching lr schedulers
        if scheduler_name == "plateau":
            scheduler = ReduceLROnPlateau(self.opt, **scheduler_args)
        elif scheduler_name == "cosineannealing":
            scheduler = CosineAnnealingLR(self.opt, **scheduler_args)
        elif scheduler_name == "cosineannealingwr":
            scheduler = CosineAnnealingWarmRestarts(self.opt,
                                                    **scheduler_args)
        elif scheduler_name == "clr":
            scheduler = CyclicLR(self.opt, **scheduler_args)
        print(f"LR Scheduler: {scheduler}")

        return scheduler

    def get_criterion(self):
        loss_name = self.criterion_params["loss"].lower()
        if loss_name == "bce_dice_loss":
            criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
        elif loss_name == "bce":
            criterion = torch.nn.BCEWithLogitsLoss()
        print(f"Criterion: {criterion}")

        return criterion

    def get_callbacks(self):
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
                      \nCheckpoints will also be saved in {resume_dir}.")
                # adding the checkpoint callback
                callbacks_list = callbacks_list + [CheckpointCallback(resume=fname,
                                                                      resume_dir=resume_dir),]
            elif mode == "model_only":
                print("Loading weights into model...")
                self.model = load_weights_train(ckpoint_params["checkpoint_path"], self.model)
        return callbacks_list

class TrainClassificationExperiment(TrainExperiment):
    """
    Stores the main parts of a classification experiment:
    - df split
    - datasets
    - loaders
    - model
    - optimizer
    - lr_scheduler
    - criterion
    - callbacks
    """
    def __init__(self, config: dict):
        """
        Args:
            config (dict): from `train_classification_yaml.py`
        """
        super().__init__(config=config)

    def get_datasets(self, train_ids, valid_ids):
        """
        Creates and returns the train and validation datasets.
        """
        # preparing transforms
        preprocessing_fn = smp.encoders.get_preprocessing_fn(self.model_params["encoder"],
                                                             "imagenet")
        preprocessing_transform = get_preprocessing(preprocessing_fn)
        train_aug = get_training_augmentation(self.io_params["aug_key"])
        val_aug = get_validation_augmentation(self.io_params["aug_key"])
        # creating the datasets
        train_dataset = ClassificationCloudDataset(self.io_params["image_folder"],
                                                   df=self.df,
                                                   im_ids=train_ids,
                                                   transforms=train_aug,
                                                   preprocessing=preprocessing_transform)
        valid_dataset = ClassificationCloudDataset(self.io_params["image_folder"],
                                                   df=self.df,
                                                   im_ids=valid_ids,
                                                   transforms=val_aug,
                                                   preprocessing=preprocessing_transform)
        return (train_dataset, valid_dataset)

    def get_model(self):
        # setting up the classification model
        model = Pretrained(variant=self.model_params["encoder"], num_classes=4,
                           pretrained=True, activation=None)
        return model

class TrainSegExperiment(TrainExperiment):
    """
    Stores the main parts of a segmentation experiment:
    - df split
    - datasets
    - loaders
    - model
    - optimizer
    - lr_scheduler
    - criterion
    - callbacks
    Note: There is no model_name for this experiment. There is `encoder` and
    `decoder` under `model_params`. You can also specify the attention_type
    """
    def __init__(self, config: dict):
        """
        Args:
            config (dict): from `train_seg_yaml.py`
        """
        super().__init__(config=config)

    def get_datasets(self, train_ids, valid_ids):
        """
        Creates and returns the train and validation datasets.
        """
        # preparing transforms
        encoder = self.model_params["encoder"]
        preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder,
                                                             "imagenet")
        preprocessing_transform = get_preprocessing(preprocessing_fn)
        train_aug = get_training_augmentation(self.io_params["aug_key"])
        val_aug = get_validation_augmentation(self.io_params["aug_key"])
        # creating the datasets
        train_dataset = CloudDataset(self.io_params["image_folder"],
                                     df=self.df,
                                     im_ids=train_ids,
                                     masks_folder=self.io_params["masks_folder"],
                                     transforms=train_aug,
                                     preprocessing=preprocessing_transform,
                                     mask_shape=self.io_params["mask_shape"])
        valid_dataset = CloudDataset(self.io_params["image_folder"],
                                     df=self.df,
                                     im_ids=valid_ids,
                                     masks_folder=self.io_params["masks_folder"],
                                     transforms=val_aug,
                                     preprocessing=preprocessing_transform,
                                     mask_shape=self.io_params["mask_shape"])
        return (train_dataset, valid_dataset)

    def get_model(self):
        encoder = self.model_params["encoder"].lower()
        decoder = self.model_params["decoder"].lower()
        print(f"\nEncoder: {encoder}, Decoder: {decoder}")
        # setting up the seg model
        assert decoder in ["unet", "fpn"], \
            "`decoder` must be one of ['unet', 'fpn']"
        if decoder == "unet":
            model = smp.Unet(encoder_name=encoder, encoder_weights="imagenet",
                             classes=4, activation=None,
                             **self.model_params[decoder])
        elif decoder == "fpn":
            model = smp.FPN(encoder_name=encoder, encoder_weights="imagenet",
                            classes=4, activation=None,
                            **self.model_params[decoder])
        # calculating # of parameters
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total # of Params: {total}\nTrainable params: {trainable}")

        return model

class TrainClfSegExperiment(TrainExperiment):
    """
    Stores the main parts of a classification + segmentation experiment:
    - df split
    - datasets
    - loaders
    - model
    - optimizer
    - lr_scheduler
    - criterion
    - callbacks
    Note: There is no model_name for this experiment. There is `encoder` and
    `decoder` under `model_params`. You can also specify the attention_type
    """
    def __init__(self, config: dict):
        """
        Args:
            config (dict): from `train_seg_yaml.py`
        """
        self.model_params = config["model_params"]
        super().__init__(config=config)

    def get_datasets(self, train_ids, valid_ids):
        """
        Creates and returns the train and validation datasets.
        """
        # preparing transforms
        encoder = self.model_params["encoder"]
        preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder,
                                                             "imagenet")
        preprocessing_transform = get_preprocessing(preprocessing_fn)
        train_aug = get_training_augmentation(self.io_params["aug_key"])
        val_aug = get_validation_augmentation(self.io_params["aug_key"])
        # creating the datasets
        train_dataset = ClfSegCloudDataset(self.io_params["image_folder"],
                                           df=self.df,
                                           im_ids=train_ids,
                                           masks_folder=self.io_params["masks_folder"],
                                           transforms=train_aug,
                                           preprocessing=preprocessing_transform,
                                           mask_shape=self.io_params["mask_shape"])
        valid_dataset = ClfSegCloudDataset(self.io_params["image_folder"],
                                           df=self.df,
                                           im_ids=valid_ids,
                                           masks_folder=self.io_params["masks_folder"],
                                           transforms=val_aug,
                                           preprocessing=preprocessing_transform,
                                           mask_shape=self.io_params["mask_shape"])
        return (train_dataset, valid_dataset)

    def get_model(self):
        encoder = self.model_params["encoder"].lower()
        decoder = self.model_params["decoder"].lower()
        print(f"\nEncoder: {encoder}, Decoder: {decoder}")
        assert encoder == "resnet34" and decoder == "fpn", \
            "Currently only ResNet34FPN is supported for CLF+Seg."
        model = ResNet34FPN(num_classes=4)
        # calculating # of parameters
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total # of Params: {total}\nTrainable params: {trainable}")

        return model

    def get_criterion(self):
        loss_dict = {
            "bce_dice_loss": BCEDiceLoss(activation="sigmoid"),
            "bce": torch.nn.BCEWithLogitsLoss(),
            "bce_no_logits": torch.nn.BCE(),
            "focal_loss": FocalLoss(logits=False),
            "heng_focal_loss": HengFocalLoss(),
        }

        seg_loss_name = self.criterion_params["seg_loss"].lower()
        clf_loss_name = self.criterion_params["clf_loss"].lower()

        # re-initializing criterion with kwargs
        seg_kwargs = self.criterion_params.get(seg_loss_name)
        clf_kwargs = self.criterion_params.get(clf_loss_name)
        seg_kwargs = {} if seg_kwargs is None else seg_kwargs
        clf_kwargs = {} if clf_kwargs is None else clf_kwargs

        seg_loss = loss_dict[seg_loss_name].__init__(**seg_kwargs)
        clf_loss = loss_dict[clf_loss_name].__init__(**clf_kwargs)

        criterion_dict = {seg_loss_name: seg_loss,
                          clf_loss_name: clf_loss}
        print(f"Criterion: {criterion_dict}")
        return criterion_dict

    def get_callbacks(self):
        from catalyst.dl.callbacks import CriterionAggregatorCallback, \
                                          CriterionCallback
        seg_loss_name = self.criterion_params["seg_loss"].lower()
        clf_loss_name = self.criterion_params["clf_loss"].lower()
        callbacks_list = [
                          CriterionCallback(prefix="seg_loss",
                                            input_key="seg_targets",
                                            output_key="seg_logits",
                                            criterion_key=seg_loss_name),
                          CriterionCallback(prefix="clf_loss",
                                            input_key="clf_targets",
                                            output_key="clf_logits",
                                            criterion_key=clf_loss_name),
                          CriterionAggregatorCallback(prefix="loss",
                                                      loss_keys=\
                                                      ["seg_loss", "clf_loss"]),
                          EarlyStoppingCallback(**self.cb_params["earlystop"]),
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
                      \nCheckpoints will also be saved in {resume_dir}.")
                # adding the checkpoint callback
                callbacks_list = callbacks_list + [CheckpointCallback(resume=fname,
                                                                      resume_dir=resume_dir),]
            elif mode == "model_only":
                print("Loading weights into model...")
                self.model = load_weights_train(ckpoint_params["checkpoint_path"], self.model)
        print(f"Callbacks: {callbacks_list}")
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
