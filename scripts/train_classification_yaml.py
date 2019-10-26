from catalyst.dl.runner import SupervisedRunner

from utils import seed_everything
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
    seed_everything(config["io_params"]["split_seed"])
    exp = TrainClassificationExperimentFromConfig(config)
    runner = SupervisedRunner()

    runner.train(
        model=exp.model,
        criterion=exp.criterion,
        optimizer=exp.opt,
        scheduler=exp.lr_scheduler,
        loaders=exp.loaders,
        callbacks=exp.cb_list,
        logdir=config["logdir"],
        num_epochs=config["num_epochs"],
        verbose=True
    )

if __name__ == "__main__":
    import yaml
    import argparse

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
