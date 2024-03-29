import os
import tqdm

from clouds.inference import Inference
from clouds.inference.utils import tta_flips_fn, apply_nonlin

class PseudoLabeler(Inference):
    def __init__(self, checkpoint_paths, test_loader, models=None,
                 mode="classification", tta_flips=None, thresh=0.8):
        """
        Attributes:
            checkpoint_paths (List[str]): Path to a checkpoint
            test_loader (torch.utils.data.DataLoader): Test loader
            models (List[None or nn.Module]): Only provide if your model weights are not traceable through torch.jit
            mode (str): either "segmentation" or "classification". Defaults to "classifcation"
            tta_flips (list-like): consisting one of or all of ["lr_flip", "ud_flip", "lrud_flip"].
                Defaults to None.
            thresh (float): threshold for hard labels
        """
        assert mode == "classification"
        self.thresh = thresh
        super().__init__(checkpoint_paths=checkpoint_paths,
                         test_loader=test_loader, models=models, mode=mode,
                         tta_flips=tta_flips)

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
                # (batch_size, n_classes)
                pred_out = apply_nonlin(self.model(test_batch[0].cuda()))
            # for each prediction (1,) in pred_out (n, 4): post process
            for pred in pred_out:
                # (4, )
                probability = pred.cpu().detach().numpy()
                for prob_i in probability:
                    # (1,)
                    predictions.append(prob_i)
        return predictions

    def create_soft_pseudo(self, sub):
        """
        Creates a dataframe with all of the probability predictions
        """
        if self.mode == "classification":
            print("Classification: Predicting classes...")
            save_path = os.path.join(os.getcwd(), "clf_probabilities.csv")
            prob_preds = self.get_classification_predictions()
        print(f"# of probs: {len(prob_preds)}")
        # Saving the submission dataframe
        sub["EncodedPixels"] = prob_preds
        sub.fillna("")
        sub.to_csv(save_path, columns=["Image_Label", "EncodedPixels"],
                   index=False)
        print(f"Saved {save_path}")
        return sub

    def create_clf_hard_pseudo(self, sub, from_soft=False):
        """
        Creates soft-pseudolabels if needed and thresholds them.
        Args:
            sub (pd.DataFrame): either the regular sample_submission.csv or
                a dataframe from self.create_soft_pseudo
            from_soft (bool): Whether or not `sub` contains the soft labels
                from self.create_soft_pseudo or not.
        """
        if not from_soft:
            sub = self.create_soft_pseudo(sub=sub)
        # thresholding to generate classification pseudolabels
        sub.loc[sub["EncodedPixels"] >= self.thresh, "EncodedPixels"] = 1
        sub.loc[sub["EncodedPixels"] < self.thresh, "EncodedPixels"] = 0
        print(f"Number of pseudo-labels: {len(sub)}")
        # saving the dataframe
        save_path = os.path.join(os.getcwd(), "clf_pseudo_labels.csv")
        sub.to_csv(save_path, columns=["Image_Label", "EncodedPixels"],
                   index=False)
        print(f"Saved the pseudo-label dataframe at {save_path}")
