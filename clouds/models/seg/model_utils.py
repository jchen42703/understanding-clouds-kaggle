import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention2d(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z

class GAB(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z

class SCse(nn.Module):
    def __init__(self, dim):
        super(SCse, self).__init__()
        self.satt = SpatialAttention2d(dim)
        self.catt = GAB(dim)

    def forward(self, x):
        return self.satt(x) + self.catt(x)

def resize_like(x, reference, mode="bilinear"):
    if x.shape[2:] !=  reference.shape[2:]:
        if mode=="bilinear":
            x = F.interpolate(x, size=reference.shape[2:],mode="bilinear",
                              align_corners=False)
        if mode=="nearest":
            x = F.interpolate(x, size=reference.shape[2:], mode="nearest")
    return x


def upsize_add(x, lateral):
    x = resize_like(x, lateral, mode="nearest") + lateral
    return x


def fuse(x, mode="cat"):
    batch_size, C0, H0, W0 = x[0].shape

    for i in range(1,len(x)):
        _,_,H,W = x[i].shape
        if (H, W)!=(H0, W0):
            x[i] = F.interpolate(x[i], size=(H0,W0), mode="bilinear",
                                 align_corners=False)
    if mode=="cat":
        return torch.cat(x, 1)


class ConvGnUp2d(nn.Module):
    def __init__(self, in_channel, out_channel, is_upsize=None, num_group=32,
                 kernel_size=3, padding=1, stride=1):
        super(ConvGnUp2d, self).__init__()
        self.is_upsize = is_upsize
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                              padding=padding, stride=stride, bias=False)
        self.gn   = nn.GroupNorm(num_group,out_channel)

    def forward(self,x):
        x = self.conv(x)
        x = self.gn(x)
        x = F.relu(x, inplace=True)
        if self.is_upsize :
            x = F.interpolate(x, scale_factor=2, mode="bilinear",align_corners=False)
        return x
