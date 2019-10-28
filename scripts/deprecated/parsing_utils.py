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

def clean_args_create_submission_no_trace(args):
    """
    Cleans up arguments for `create_submission_no_trace.py` where:
    args = parser.parse_args()
    """
    # cleaning up --classification_models
    # making it so that single element lists are considered as a single
    # model (so there won't be redundant averaging from ensembling)
    args.clf_model_names = args.clf_model_names[0] \
                                    if len(args.clf_model_names)==1 \
                                    else args.clf_model_names
    # cleaning up --checkpoint_paths
    # to be consistent with the args.classification_models' reason
    args.checkpoint_paths = args.checkpoint_paths[0] \
                                    if len(args.checkpoint_paths)==1 \
                                    else args.checkpoint_paths
    if isinstance(args.checkpoint_paths, list) and isinstance(args.clf_model_names, list):
        assert len(args.checkpoint_paths) == len(args.clf_model_names), \
            "There must be the same number of checkpoint paths as the number of \
            models specified."
    # cleaning up --tta
    if isinstance(args.tta, str):
        # handles both the "None" case and the single TTA op case
        # --tta="None" or --tta="..."
        args.tta = [] if args.tta == "None" else [args.tta]
    elif args.tta == ["None"]:
        # handles case where --tta "None"
        args.tta = []
    return args
