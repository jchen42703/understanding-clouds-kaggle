from abc import abstractmethod
import os

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from clouds.io import ClassificationCloudDataset, CloudDataset, \
                      ClfSegCloudDataset
from .utils import setup_train_and_sub_df, get_validation_augmentation, \
                   get_preprocessing

class InferExperiment(object):
    def __init__(self, config: dict):
        """
        Args:
            config (dict):

        Attributes:
            config-related:
                config (dict):
                io_params (dict):
                    in_dir (key: str): path to the data folder
                    test_size (key: float): split size for test
                    split_seed (key: int): seed
                    batch_size (key: int): <-
                    num_workers (key: int): # of workers for data loaders
            split_dict (dict): test_ids
            test_dset (torch.data.Dataset): <-
            loaders (dict): train/validation loaders
            model (torch.nn.Module): <-
        """
        # for reuse
        self.config = config
        self.io_params = config["io_params"]
        self.model_params = config["model_params"]
        # initializing the experiment components
        self.df, self.sample_sub, _ = self.setup_df()
        test_ids = self.get_test_ids()
        self.test_dset = self.get_datasets(test_ids)
        self.loaders = self.get_loaders()
        self.model = self.get_model()

    @abstractmethod
    def get_datasets(self, test_ids):
        """
        Initializes the data augmentation and preprocessing transforms. Creates
        and returns the train and validation datasets.
        """
        return

    @abstractmethod
    def get_models(self):
        """
        Creates and returns the models to infer (and ensemble). Note that
        this differs from TrainExperiment variants because they only fetch
        one model.
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

    def get_test_ids(self):
        """
        Returns the test image ids.
        """
        image_labels = self.sample_sub["Image_Label"]
        test_ids = image_labels.apply(lambda x: x.split("_")[0]).drop_duplicates().values
        print(f"Number of test ids: {len(test_ids)}")
        n_encoded = len(sub["EncodedPixels"])
        print(f"length of sub: {n_encoded}")
        return test_ids

    def get_loaders(self):
        """
        Creates train/val loaders from datasets created in self.get_datasets.
        Returns the loaders.
        """
        # setting up the loaders
        b_size, num_workers = self.io_params["batch_size"], self.io_params["num_workers"]
        test_loader = DataLoader(self.test_dset, batch_size=b_size,
                                  shuffle=False, num_workers=num_workers)
        return {"test": test_loader}

class GeneralInferExperiment(InferExperiment):
    def __init__(self, config: dict):
        """
        Args:
            config (dict):

        Attributes:
            config-related:
                config (dict):
                io_params (dict):
                    in_dir (key: str): path to the data folder
                    test_size (key: float): split size for test
                    split_seed (key: int): seed
                    batch_size (key: int): <-
                    num_workers (key: int): # of workers for data loaders
            split_dict (dict): test_ids
            test_dset (torch.data.Dataset): <-
            loaders (dict): train/validation loaders
            model (torch.nn.Module): <-
        """
        super().__init__(config=config)
        self.mode = self.config["mode"]
        self.encoders = self.config["encoders"]
        self.decoders = self.config["decoders"]

    def get_datasets(self, test_ids):
        preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoders[0],
                                                             "imagenet")
        preprocessing_transform = get_preprocessing(preprocessing_fn)
        val_aug = get_validation_augmentation(self.io_params["aug_key"])
        # fetching the proper datasets and models
        print("Assuming that all encoders are from the same family...")
        if self.mode == "segmentation" or self.mode == "both":
            test_dataset = CloudDataset(self.io_params["image_folder"], df=sub,
                                        im_ids=test_ids,
                                        transforms=val_aug,
                                        preprocessing=preprocessing_transform)
        elif self.mode == "classification":
            test_dataset = ClassificationCloudDataset(self.io_params["image_folder"],
                                                      df=sub, im_ids=test_ids,
                                                      transforms=val_aug,
                                                      preprocessing=preprocessing_transform)
        return test_dataset

    def get_models(self):
        """
        Fetches multiple models as a list. If it's a single model, `models`
        will be a length one list.
        """
        if self.mode == "segmentation":
            pairs = list(zip(self.encoders, self.decoders))
            print(f"Models: {pairs}")
            # setting up the seg model
            models = [smp.__dict__[decoder](encoder_name=encoder,
                                            encoder_weights=None,
                                            classes=4, activation=None,
                                            **self.model_params[decoder])
                      for encoder, decoder in pairs]
        elif self.mode == "classification":
            models = [Pretrained(variant=name, num_classes=4, pretrained=False)
                      for name in self.encoders]
        elif self.mode == "both":
            print("Currently only supporting: clouds.models.ResNet34FPN")
            models = [ResNet34FPN(num_classes=4, do_inference=True)
                      for name in self.encoders]
        return models
