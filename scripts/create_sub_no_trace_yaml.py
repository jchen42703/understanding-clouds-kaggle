import gc
import torch

from clouds.inference import Inference
from clouds.experiments import GeneralInferExperiment

def main(config):
    """
    Main code for creating the segmentation-only submission file. All masks are
    converted to either "" or RLEs

    Args:
        args (instance of argparse.ArgumentParser): arguments must be compiled with parse_args
    Returns:
        None
    """
    torch.cuda.empty_cache()
    gc.collect()

    exp = GeneralInferExperiment(config)
    infer = Inference(config["checkpoint_paths"], exp.loaders["test"],
                      models=exp.models, mode=exp.mode,
                      **config["infer_params"])
    out_df = infer.create_sub(sub=exp.sample_sub)

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
