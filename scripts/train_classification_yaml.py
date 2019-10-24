from catalyst.dl.runner import SupervisedRunner

from utils import setup_train_and_sub_df, seed_everything
from experiment import TrainClassificationExperimentFromConfig

def main(config):
    """
    Main code for training a classification model.

    Args:
        config (dict): dictionary read from a yaml file
            i.e. experiments/finetune_classification.yml
    Returns:
        None
    """
    # setting up the train/val split with filenames
    train, sub, id_mask_count = setup_train_and_sub_df(config["dset_path"])
    # setting up the train/val split with filenames
    seed_everything(config["io_params"]["split_seed"])
    exp = TrainClassificationExperimentFromConfig(config, train, id_mask_count)
    runner = SupervisedRunner()

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

if __name__ == "__main__":
    import yaml
    import argparser

    parser = argparse.ArgumentParser(description="For training.")
    parser.add_argument("--yml_path", type=str, required=True,
                        help="Path to the .yml config.")
    args = parser.parse_args()

    with open(args.yml_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    main(config)
