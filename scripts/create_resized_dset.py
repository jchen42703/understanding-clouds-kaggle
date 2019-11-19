from clouds import Preprocessor
from clouds.experiments import setup_train_and_sub_df

def main(config):
    paths_params = config["paths_params"]
    paths_dict = {
        "train_dir": paths_params["train_dir"],
        "test_dir": paths_params["test_dir"],
        "train_out": paths_params["train_out"],
        "test_out": paths_params["test_out"],
        "mask_out": paths_params["mask_out"],
    }
    train, sub, _ = setup_train_and_sub_df(paths_params["train_csv_path"],
                                           paths_params["sample_sub_csv_path"])
    preprocessor = Preprocessor(train, paths_dict, tuple(config["out_shape_cv2"]),
                                config["file_type"])
    if config["process_train_test"]:
        preprocessor.execute_train_test()
    if config["process_masks"]:
        preprocessor.execute_masks()

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
