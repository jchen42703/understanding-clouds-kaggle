import numpy as np
import cv2
import tqdm
import torch
import tqdm
import os
import pandas as pd

from functools import partial
from torch.jit import load
from clouds.inference.utils import mask2rle, post_process, load_weights_infer, \
                                  tta_flips_fn, apply_nonlin

class Inference(object):
    def __init__(self, checkpoint_paths, test_loader, test_dataset, models=None,
                 mode="segmentation", tta_flips=None):
        """
        Attributes:
            checkpoint_paths (List[str]): Path to a checkpoint
            test_loader (torch.utils.data.DataLoader): Test loader
            models (List[None or nn.Module]): Only provide if your model weights are not traceable through torch.jit
            mode (str): either "segmentation" or "classification". Defaults to "segmentation"
            tta_flips (list-like): consisting one of or all of ["lr_flip", "ud_flip", "lrud_flip"].
                Defaults to None.
        """
        self.load_checkpoints(checkpoint_paths, models)

        self.mode = mode
        self.loader = test_loader
        self.dataset = test_dataset
        self.seg_class_params = {0: (0.5, 10000), 1: (0.5, 10000), 2: (0.5, 10000),
                                 3: (0.5, 10000)} # (threshold, min_size)
        self.tta_fn = None
        if tta_flips is not None:
            assert isinstance(tta_flips, (list, tuple)), \
                "tta_flips must be a list-like of strings."
            print(f"TTA Ops: {tta_flips}")
            self.tta_fn = partial(tta_flips_fn, model=self.model, mode=mode,
                                  flips=tta_flips, non_lin="sigmoid")

    def load_checkpoints(self, checkpoint_paths, models):
        """
        Either loads a single checkpoint or loads an ensemble of checkpoints
        from `checkpoint_paths`
        """
        # single model instances
        if isinstance(checkpoint_paths, str):
            try:
                self.model = load(checkpoint_paths).cuda().eval()
                print(f"Traced model from {checkpoint_paths}")
            except:
                self.model = load_weights_infer(checkpoint_paths, models).cuda().eval()
                print(f"Loaded model from {checkpoint_paths}")
        # ensembled models
        elif len(checkpoint_paths) > 1:
            try:
                self.model = EnsembleModel([load(path).cuda().eval()
                                            for path in checkpoint_paths])
            except:
                self.model = EnsembleModel([load_weights_infer(path, model).cuda().eval()
                                            for (path, model) in
                                            zip(checkpoint_paths, models)])
            print(f"Loaded an ensemble from {checkpoint_paths}")

    def create_sub(self, sub):
        """
        Creates and saves a submission dataframe (classification/segmentation).
        Args:
            sub (pd.DataFrame): the same sub used for the test_dataset; the sample_submission dataframe (stage1).
                This is used to create the final submission dataframe
        Returns:
            submission (pd.DataFrame): submission dataframe
        """
        if self.mode == "segmentation":
            print("Segmentation: Converting predicted masks to run-length-encodings...")
            save_path = os.path.join(os.getcwd(), "submission.csv")
            encoded_pixels = self.get_encoded_pixels()
        elif self.mode == "classification":
            print("Classification: Predicting classes...")
            save_path = os.path.join(os.getcwd(), "submission_classification.csv")
            encoded_pixels = self.get_classification_predictions()
        # Saving the submission dataframe
        sub["EncodedPixels"] = encoded_pixels
        sub.fillna("")
        sub.to_csv(save_path, columns=["ImageId_ClassId", "EncodedPixels"], index=False)
        print(f"Saved the submission file at {save_path}")
        return sub

    def get_encoded_pixels(self):
        """
        Processes predicted logits and converts them to encoded pixels. Does so in an iterative
        manner so operations are done image-wise rather than on the full dataset directly (to
        combat RAM limitations).

        Returns:
            encoded_pixels: list of rles in the order of self.loader
        """
        encoded_pixels = []
        image_id = 0
        for i, test_batch in enumerate(tqdm.tqdm(self.loader)):
            if self.tta_fn is not None:
                pred_out = self.tta_fn(batch=test_batch[0].cuda())
            else:
                pred_out = apply_nonlin(self.model(test_batch[0].cuda()))
            # for each batch (4, h, w): resize and post_process
            for i, batch in enumerate(pred_out):
                for probability in batch:
                    # iterating through each probability map (h, w)
                    probability = probability.cpu().detach().numpy()
                    if probability.shape != (256, 1600):
                        probability = cv2.resize(probability, dsize=(1600, 256), interpolation=cv2.INTER_LINEAR)
                    predict, num_predict = post_process(probability, self.seg_class_params[image_id % 4][0],
                                                        self.seg_class_params[image_id % 4][1])
                    if num_predict == 0:
                        encoded_pixels.append("")
                    else:
                        r = mask2rle(predict)
                        encoded_pixels.append(r)
                    image_id += 1
        return encoded_pixels

    def get_classification_predictions(self):
        """
        Processes predicted logits and converts them to encoded pixels. Does so in an iterative
        manner so operations are done image-wise rather than on the full dataset directly (to
        combat RAM limitations).

        Returns:
            List of predictions ("" if 0 and "1" if 1)
        """
        predictions = []
        for i, test_batch in enumerate(tqdm.tqdm(self.loader)):
            if self.tta_fn is not None:
                pred_out = self.tta_fn(batch=test_batch[0].cuda())
            else:
                pred_out = apply_nonlin(self.model(test_batch[0].cuda()))
            # for each batch (n, 4): post process
            for i, batch in enumerate(pred_out):
                # iterating through each prediction (4,)
                probability = batch.cpu().detach().numpy()

                predict = cv2.threshold(probability, 0.5, 1, cv2.THRESH_BINARY)[1]
                # Idea: [imgid_1, imgid_2, imgid_3, imgid_4, imgid2_1,...]
                def process(element):
                    if element == 0:
                        return ""
                    else:
                        return "1 1"
                predict = list(map(process, predict.flatten().tolist()))
                predictions = predictions + predict
        return predictions

class EnsembleModel(object):
    """
    Callable class for ensembled model inference
    """
    def __init__(self, models):
        self.models = models
        assert len(self.models) > 1

    def __call__(self, x):
        res = []
        with torch.no_grad():
            for m in self.models:
                res.append(m(x))
        res = torch.stack(res)
        return torch.mean(res, dim=0)
