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
