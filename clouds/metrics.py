import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torch.nn as nn

class BCEDiceLoss(smp.utils.losses.DiceLoss):
    def __init__(self, eps=1e-7, activation="sigmoid"):
        super().__init__(eps, activation)
        if activation is None or activation == "none":
            # activation was applied beforehand by the NN
            self.bce = nn.BCE(reduction="mean")
        else:
            self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return dice + bce

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class HengFocalLoss(nn.Module):
    """
    From Heng's starter kit:
    https://www.kaggle.com/c/understanding_cloud_organization/discussion/115787#latest-674710
    Assumes that the model returns probabilities not logits! Also, not tested
    for segmentation.
    """
    def __init__(self):
        super(HengFocalLoss, self).__init__()
        print("Assumes the model returns probabilities, not logits!")

    def forward(self, inputs, targets):
        # clipping probabilities
        p = torch.clamp(inputs, 1e-9, 1-1e-9)
        loss_label = -targets*torch.log(p) - 2*(1-targets)*torch.log(1-p)
        loss_label = loss_label.mean()
        return loss_label
