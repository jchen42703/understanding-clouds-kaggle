import albumentations as albu
import pandas as pd

from clouds.io.utils import to_tensor

def setup_train_and_sub_df(path):
    """
    Sets up the training and sample submission DataFrame.
    Args:
        path (str): Base diretory where train.csv and sample_submission.csv are located
    Returns:
        tuple of:
            train (pd.DataFrame): The prepared training dataframe with the extra columns:
                im_id & label
            sub (pd.DataFrame): The prepared sample submission dataframe with the
                same extra columns as train
    """
    # Reading the in the .csvs
    train = pd.read_csv(os.path.join(path, "train.csv"))
    sub = pd.read_csv(os.path.join(path, "sample_submission.csv"))

    # setting the dataframe for training/inference
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])

    sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
    sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])
    id_mask_count = train.loc[train["EncodedPixels"].isnull() == False, "Image_Label"].apply(lambda x: x.split("_")[0]).value_counts().\
    reset_index().rename(columns={"index": "im_id", "Image_Label": "count"})
    return (train, sub)

def get_training_augmentation(use_resized_dataset=False):
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
    ]
    if not use_resized_dataset:
        train_transform = train_transform.append(albu.Resize(320, 640))
    return albu.Compose(train_transform)


def get_validation_augmentation(use_resized_dataset=False):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
    ]
    if not use_resized_dataset:
        test_transform.append(albu.Resize(320, 640))
    return albu.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
