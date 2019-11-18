import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import SCse, ConvGnUp2d, fuse, upsize_add

class ResNet34FPN(nn.Module):
    """
    FPN with a pretrained ResNet34 encoder. Returns both the classification
    label and the segmentation label
    """
    def __init__(self, num_classes=4, do_inference=False, fp16: bool = False):
        super(ResNet34FPN, self).__init__()
        if not do_inference:
            print("This model returns probabilities, not logits! Please",
                  "make sure that you have the appropriate criterion selected.")
        self.infer = do_inference
        self.fp16 = fp16
        pretrained = False if self.infer else True

        self.resnet = torchvision.models.resnet34(pretrained=pretrained)
        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu)

        self.encode2 = nn.Sequential(self.resnet.layer1,
                                     SCse(64))
        self.encode3 = nn.Sequential(self.resnet.layer2,
                                     SCse(128))
        self.encode4 = nn.Sequential(self.resnet.layer3,
                                     SCse(256))
        self.encode5 = nn.Sequential(self.resnet.layer4,
                                     SCse(512))
        #---
        self.lateral0 = nn.Conv2d(512, 128, kernel_size=1, padding=0, stride=1)
        self.lateral1 = nn.Conv2d(256, 128, kernel_size=1, padding=0, stride=1)
        self.lateral2 = nn.Conv2d(128, 128, kernel_size=1, padding=0, stride=1)
        self.lateral3 = nn.Conv2d(64, 128, kernel_size=1, padding=0, stride=1)

        self.top1 = nn.Sequential(
            ConvGnUp2d(128,128, True),
            ConvGnUp2d(128, 64, True),
            ConvGnUp2d( 64, 64, True),
        )
        self.top2 = nn.Sequential(
            ConvGnUp2d(128, 64, True),
            ConvGnUp2d(64, 64, True),
        )
        self.top3 = nn.Sequential(
            ConvGnUp2d(128, 64, True),
        )
        self.logit = nn.Conv2d(64*3, num_classes, kernel_size=1)


    def forward(self, x):
        batch_size, C, H, W = x.shape

        x0 = self.conv1(x)
        x1 = self.encode2(x0)
        x2 = self.encode3(x1)
        x3 = self.encode4(x2)
        x4 = self.encode5(x3)

        ##----
        #segment

        t0 = self.lateral0(x4)
        t1 = upsize_add(t0, self.lateral1(x3)) #16x16
        t2 = upsize_add(t1, self.lateral2(x2)) #32x32
        t3 = upsize_add(t2, self.lateral3(x1)) #64x64

        t1 = self.top1(t1) #; print(t1.shape)
        t2 = self.top2(t2) #; print(t2.shape)
        t3 = self.top3(t3) #; print(t3.shape)

        x = fuse([t1, t2, t3], "cat")
        logit = self.logit(x)

        #---
        probability_mask = torch.sigmoid(logit)
        probability_label = F.adaptive_max_pool2d(probability_mask, 1).view(batch_size,-1)
        if self.infer:
            return logit
        elif self.fp16:
            # amp doesn't support regular BCELoss; only BCEWithLogitsLoss
            return (probability_label, logit)
        else:
            return (probability_label, probability_mask)
