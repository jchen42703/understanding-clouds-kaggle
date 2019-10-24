import os

from clouds.inference import Inference
from clouds.inference.utils import tta_flips_fn, apply_nonlin

class PseudoLabeler(Inference):
    def __init__(self, checkpoint_paths, test_loader, test_dataset, models=None,
                 mode="classification", tta_flips=None, thresh_lower=0.2,
                 thresh_upper=0.8):
        """
        Attributes:
            checkpoint_paths (List[str]): Path to a checkpoint
            test_loader (torch.utils.data.DataLoader): Test loader
            models (List[None or nn.Module]): Only provide if your model weights are not traceable through torch.jit
            mode (str): either "segmentation" or "classification". Defaults to "classifcation"
            tta_flips (list-like): consisting one of or all of ["lr_flip", "ud_flip", "lrud_flip"].
                Defaults to None.
        """
        assert mode == "classification"
        self.thresh_lower = thresh_lower
        self.thresh_upper = thresh_upper
        super().__init__(checkpoint_paths=checkpoint_paths,
                         test_loader=test_loader, test_dataset=test_dataset,
                         models=models, mode=mode, tta_flips=tta_flips)

    def get_encoded_pixels(self):
        raise NotImplementedError

    def get_classification_predictions(self):
        """
        Gets the raw classification predictions

        Returns:
            List of probability predictions
        """
        predictions = []
        for i, test_batch in enumerate(tqdm.tqdm(self.loader)):
            if self.tta_fn is not None:
                pred_out = self.tta_fn(batch=test_batch[0].cuda())
            else:
                pred_out = apply_nonlin(self.model(test_batch[0].cuda()))
            # for each prediction (1,) in pred_out (n, 4): post process
            for pred in pred_out:
                probability = pred.cpu().detach().numpy()
                predictions.append(probability)
        return predictions

    def create_prob_pred_df(self):
        """
        Creates a dataframe with all of the probability predictions
        """
        if self.mode == "classification":
            print("Classification: Predicting classes...")
            save_path = os.path.join(os.getcwd(), "clf_probabilities.csv")
            prob_preds = self.get_classification_predictions()
        # Saving the submission dataframe
        sub["EncodedPixels"] = prob_preds
        sub.fillna("")
        sub.to_csv(save_path, columns=["ImageId_ClassId", "EncodedPixels"],
                   index=False)
        print(f"Saved {save_path}")
        return sub

    def create_clf_pseudo_df(self):
        """
        Saves the classification pseudolabels in a dataframe.
        * creates regular dataframe with all confidence values
        * threshold them
        * set the rest as NaN and drop all NaNs
        * save the dataframe
        """
        sub = self.create_prob_pred_df()
        print(f"Number of predictions before: {len(sub)}")
        # thresholding to generate classification pseudolabels
        sub[sub["EncodedPixels"] >= self.thresh_upper] = 1
        sub[sub["EncodedPixels"] <= self.thresh_lower] = 0
        # getting rid of non-pseudo-labels
        sub[sub["EncodedPixels"] != 0 and sub["EncodedPixels"] != 1] = None
        sub = sub.dropna()
        print(f"Number of pseudo-labels: {len(sub)}")
        # saving the dataframe
        save_path = os.path.join(os.getcwd(), "clf_pseudo_labels.csv")
        sub.to_csv(save_path, columns=["ImageId_ClassId", "EncodedPixels"],
                   index=False)
        print(f"Saved the pseudo-label dataframe at {save_path}")
