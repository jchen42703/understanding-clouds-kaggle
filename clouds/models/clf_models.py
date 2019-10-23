import torch.nn as nn
import torch

import pretrainedmodels

class Pretrained(nn.Module):
    """
    A generalized class for fetching a pretrained model from Cadene/pretrainedmodels
    From: https://github.com/catalyst-team/mlcomp/blob/master/mlcomp/contrib/model/pretrained.py
    """
    def __init__(self, variant, num_classes, pretrained=True, activation=None):
        super().__init__()
        params = {'num_classes': 1000}
        if not pretrained:
            params['pretrained'] = None
        model = pretrainedmodels.__dict__[variant](**params)
        if "se_res" in variant:
            model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.need_refactor = False
        if 'resnet' in variant:
            self.need_refactor = True

        if self.need_refactor:
            self.l1 = nn.Sequential(*list(model.children())[:-1])
            if torch.cuda.is_available():
                self.l1 = self.l1.to('cuda:0')
            self.last = nn.Linear(model.last_linear.in_features, num_classes)
        else:
            self.model = model
            linear = self.model.last_linear
            if isinstance(linear, nn.Linear):
                self.model.last_linear = nn.Linear(
                    model.last_linear.in_features,
                    num_classes
                )
            elif isinstance(linear, nn.Conv2d):
                self.model.last_linear = nn.Conv2d(
                    linear.in_channels,
                    num_classes,
                    kernel_size=linear.kernel_size,
                    bias=True
                )

        if callable(activation) or activation is None:
            self.activation = activation
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(
                'Activation should be "sigmoid"/"softmax"/callable/None')

    def forward(self, x):
        if not self.need_refactor:
            res = self.model(x)
            if isinstance(res, tuple):
                return res[0]
            return res
        x = self.l1(x)
        x = x.view(x.size()[0], -1)
        x = self.last(x)
        if self.activation:
            x = self.activation(x)
        return x

__all__ = ["Pretrained"]
