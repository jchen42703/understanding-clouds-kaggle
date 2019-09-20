import os
import numpy as np
import pandas as pd
import cv2
import segmentation_models_pytorch as smp
import tqdm
import pickle

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from catalyst.dl.runner import SupervisedRunner

from clouds.io.dataset import CloudDataset
from clouds.io.utils import post_process, sigmoid
from clouds.metrics import dice
from utils import get_preprocessing, get_validation_augmentation, setup_train_and_sub_df

def main(path="../input/understanding_cloud_organization", bs=5, encoder="resnet50",
         test_size=0.1, use_resized_dataset=False, split_seed=42, attention_type="scse"):
    """
    Runs `predict_validation` and `get_class_params` and saves the output `class_params`
    """
    probabilities, valid_masks = predict_validation(path=path, bs=bs, encoder=encoder,
                                                    test_size=test_size, use_resized_dataset=use_resized_dataset,
                                                    split_seed=split_seed, attention_type=attention_type)
    class_params = get_class_params(probabilities, valid_masks)
    save_path = os.path.join(path, "class_params.pickle")
    print(f"Saving class params at {save_path}")
    with open(save_path, "wb") as handle:
        pickle.dump(class_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

def predict_validation(path="../input/understanding_cloud_organization", bs=5, encoder="resnet50",
                       test_size=0.1, use_resized_dataset=False, split_seed=42, attention_type="scse"):
    """
    Predicts the validation set; assumes the model weights are in ./logdir/checkpoints/best.pth.
    """
    # Reading the in the .csvs
    train = pd.read_csv(os.path.join(path, "train.csv"))
    sub = pd.read_csv(os.path.join(path, "sample_submission.csv"))

    # setting up the train/val split with filenames
    train, sub, id_mask_count = setup_train_and_sub_df(path)
    # setting up the train/val split with filenames
    train_ids, valid_ids = train_test_split(id_mask_count["im_id"].values, random_state=split_seed,
                                            stratify=id_mask_count["count"], test_size=test_size)
    # setting up model (U-Net with random weight initializations and logit outputs)
    DEVICE = "cuda"
    attention_type = None if attention_type == "None" else attention_type
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        classes=4,
        activation=None,
        attention_type=attention_type
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, "imagenet")
    # Setting up the I/O
    num_workers = 0
    valid_dataset = CloudDataset(path, df=train, datatype="valid", im_ids=valid_ids,
                                 transforms=get_validation_augmentation(use_resized_dataset), preprocessing=get_preprocessing(preprocessing_fn),
                                 use_resized_dataset=use_resized_dataset)

    valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

    # everything is saved here (i.e. weights + stats)
    logdir = "./logs/segmentation"

    loaders = {"valid": valid_loader}

    ckpoint_path = os.path.join(logdir, "checkpoints", "best.pth")
    runner = SupervisedRunner()
    predictions = runner.predict_loader(
        model=model,
        loader=loaders["valid"],
        resume=ckpoint_path
    )

    valid_masks = []
    probabilities = np.zeros((2220, 350, 525))
    for i, (batch, output) in enumerate(tqdm.tqdm(zip(
            valid_dataset, predictions))):
        image, mask = batch
        for m in mask:
            if m.shape != (350, 525):
                m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            valid_masks.append(m)

        for j, probability in enumerate(output):
            if probability.shape != (350, 525):
                probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            probabilities[i * 4 + j, :, :] = probability
    return (probabilities, valid_masks)

def get_class_params(probabilities, valid_masks):
    """
    Finds the best threshold and min_roi_size for each class and stores them in a dict
    """
    class_params = {}
    for class_id in range(4):
        print(class_id)
        attempts = []
        for t in range(0, 100, 5):
            t /= 100
            for ms in [0, 100, 1200, 5000, 10000]:
                masks = []
                for i in range(class_id, len(probabilities), 4):
                    probability = probabilities[i]
                    predict, num_predict = post_process(sigmoid(probability), t, ms)
                    masks.append(predict)

                d = []
                for i, j in zip(masks, valid_masks[class_id::4]):
                    if (i.sum() == 0) & (j.sum() == 0):
                        d.append(1)
                    else:
                        d.append(dice(i, j))

                attempts.append((t, ms, np.mean(d)))

        attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])

        # Getting the best threshold and min size
        attempts_df = attempts_df.sort_values('dice', ascending=False)
        print(attempts_df.head())
        best_threshold = attempts_df['threshold'].values[0]
        best_size = attempts_df['size'].values[0]

        class_params[class_id] = (best_threshold, best_size)
    print(f"Class Parameters: {class_params}")
    return class_params

def add_bool_arg(parser, name, default=False):
    """
    From: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    Handles boolean cases from command line through the creating two mutually exclusive arguments: --name and --no-name.
    Args:
        parser (arg.parse.ArgumentParser): the parser you want to add the arguments to
        name: name of the common feature name for the two mutually exclusive arguments; dest = name
        default: default boolean for command line
    Returns:
        None
    """
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=name, action="store_true")
    group.add_argument("--no-" + name, dest=name, action="store_false")
    parser.set_defaults(**{name:default})

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
    parser.add_argument("--test_size", type=float, required=False, default=0.1,
                        help="Fraction of total dataset to make the validation set.")
    add_bool_arg(parser, "use_resized_dataset", default=False)
    parser.add_argument("--split_seed", type=int, required=False, default=42,
                        help="Seed for the train/val dataset split")
    parser.add_argument("--attention_type", type=str, required=False, default="scse",
                        help="Attention type; if you want None, just put the string None.")
    args = parser.parse_args()
    main(path=args.dset_path, bs=args.batch_size, encoder=args.encoder,
         test_size=args.test_size, use_resized_dataset=args.use_resized_dataset,
         split_seed=args.split_seed, attention_type=args.attention_type)
