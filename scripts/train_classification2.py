import torch

from pathlib import Path
from catalyst.dl.callbacks import DiceCallback, AccuracyCallback, \
                                  EarlyStoppingCallback, CheckpointCallback
from catalyst.dl.runner import SupervisedRunner

from clouds.custom.ppv_tpr_f1 import PrecisionRecallF1ScoreCallback
from utils import setup_train_and_sub_df, seed_everything
from experiment import TrainClassificationExperiment

def main(args):
    """
    Main code for training a classification model.

    Args:
        args (instance of argparse.ArgumentParser): arguments must be compiled with parse_args
    Returns:
        None
    """
    # setting up the train/val split with filenames
    train, sub, id_mask_count = setup_train_and_sub_df(args.dset_path)
    # setting up the train/val split with filenames
    seed_everything(args.split_seed)
    exp = TrainClassificationExperiment(args, train, id_mask_count)
    runner = SupervisedRunner()

    callbacks_list = [PrecisionRecallF1ScoreCallback(num_classes=4),#DiceCallback(),
                      EarlyStoppingCallback(patience=5, min_delta=0.001),
                      AccuracyCallback(threshold=0.5, activation="Sigmoid"),
                      ]

    if args.checkpoint_path != "None": # hacky way to say no checkpoint callback but eh what the heck
        if args.checkpoint_mode.lower() == "full":
            print("Stateful loading...")
            ckpoint_p = Path(args.checkpoint_path)
            fname = ckpoint_p.name
            resume_dir = str(ckpoint_p.parents[0]) # everything in the path besides the base file name
            print(f"Loading {fname} from {resume_dir}. Checkpoints will also be saved in {resume_dir}.")
            callbacks_list = callbacks_list + [CheckpointCallback(resume=fname, resume_dir=resume_dir),]
        elif args.checkpoint_mode.lower() == "model_only":
            print("Loading weights into model...")
            model = load_weights_train(args.checkpoint_path, model)

    runner.train(
        model=exp.model,
        criterion=exp.criterion,
        optimizer=exp.optimizer,
        scheduler=exp.scheduler,
        loaders=exp.loaders,
        callbacks=callbacks_list,
        logdir=logdir,
        num_epochs=args.num_epochs,
        verbose=True
    )

def load_weights_train(checkpoint_path, model):
    """
    Loads py model from a checkpoint and into inference mode.

    Args:
        checkpoint_path (str): path to a .pt or .pth checkpoint
        model (torch.nn.Module): <-
    Returns:
        Model with loaded weights and in evaluation mode
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

if __name__ == "__main__":
    import argparse
    # parsing the arguments from the command prompt
    parser = argparse.ArgumentParser(description="For training.")
    parser.add_argument("--dset_path", type=str, required=True,
                        help="Path to the unzipped kaggle dataset directory.")
    parser.add_argument("--model_name", type=str, required=False, default="resnet50",
                        help="one of the models in https://github.com/Cadene/pretrained-models.pytorch")
    parser.add_argument("--num_epochs", type=int, required=False, default=21,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, required=False, default=16,
                        help="Batch size")
    parser.add_argument("--df_setup_type", type=str, required=False, default="regular",
                        help="`regular` or `pos_only` for how the ids are set up.")
    parser.add_argument("--test_size", type=float, required=False, default=0.1,
                        help="Fraction of total dataset to make the validation set.")
    parser.add_argument("--split_seed", type=int, required=False, default=42,
                        help="Seed for the train/val dataset split")
    parser.add_argument("--num_workers", type=int, required=False, default=2,
                        help="Number of workers for data loaders.")
    parser.add_argument("--loss", type=str, required=False, default="bce",
                        help="Either bce_dice_loss or bce")
    parser.add_argument("--lr", type=float, required=False, default=3e-4,
                        help="Learning rate.")
    parser.add_argument("--scheduler", type=str, required=False, default="plateau",
                        help="Learning rate scheduler; one of 'clr', \
                        'plateau', 'cosineannealing', or 'cosineannealingwr'.")
    parser.add_argument("--checkpoint_path", type=str, required=False, default="None",
                        help="Checkpoint path; if you want to train from scratch, just put the string as None.")
    parser.add_argument("--checkpoint_mode", type=str, required=False, default="full",
                        help="'full' for stateful loading or 'model_only' for just loading weights.")
    parser.add_argument("--opt", type=str, required=False, default="adam",
                        help="Optimizer")
    parser.add_argument("--aug_key", type=str, required=False, default="aug4",
                        help="Augmentation key for get_training_augmentation")
    args = parser.parse_args()

    main(args)
