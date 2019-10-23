from clouds import Preprocessor

def main(args):
    train, sub, _ = setup_train_and_sub_df(args.dset_path)
    COLAB_PATHS_DICT = {
        "train_dir": "./train_images/"
        "test_dir": "./test_images/"
        "train_out": "train640.zip"
        "test_out": "test640.zip"
        "mask_out": "masks640.zip"
    }
    preprocessor = Preprocessor(train, COLAB_PATHS_DICT, args.out_shape_cv2,
                                args.file_type)
    if args.process_train_test:
        preprocessor.execute_train_test()
    if args.process_masks:
        preprocessor.execute_masks()

if __name__ == "__main__":
    import argparse
    from parsing_utils import add_bool_arg

    # parsing the arguments from the command prompt
    parser = argparse.ArgumentParser(description="For creating the resized dataset.")
    parser.add_argument("--dset_path", type=str, required=True,
                        help="Path to the unzipped kaggle dataset directory.")
    add_bool_arg(parser, "process_train_test", default=True)
    add_bool_arg(parser, "process_masks", default=True)
    parser.add_argument("--file_type", type=str, required=False, default=".jpg",
                        help="file type")
    parser.add_argument("--out_shape_cv2", n_args="+", type=str, required=False,
                        default=[640, 320], help="Shape to resize to in cv2")



    args = parser.parse_args()
    main(args)
