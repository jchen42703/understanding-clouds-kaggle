# Understanding Clouds from Satellite Images

## Approach:
* Segmentation: seresnext50_unet
* Classification to Remove FPs
  * `pretrainedmodels` pseudolabels
    * resnet34
    * se_resnext50_32x4d
    * inceptionv4
  * resnet34 (classification + bbox prediction)


## Needed Libraries
(from: https://www.kaggle.com/artgor/segmentation-in-pytorch-using-convenient-tools)
* Pytorch (torchvision==0.4)
* catalyst
* pytorch_toolbelt
* pretrainedmodels
* https://github.com/qubvel/segmentation_models.pytorch
