# Understanding Clouds from Satellite Images
Using convolutional neural networks for the segmentation of cloud structures from satellite images in [this kaggle challenge](https://www.kaggle.com/c/understanding_cloud_organization).

## Approach:
* __Current Best Score (4 models, 264th Place):__ Public LB: `0.6608`, Private LB: `0.65019`
* __Segmentation:__ `seresnext50_unet/seresnext50_fpn/efficientnetb4_unet`
* __Classification to Remove FPs:__ `se_resnext50_32x4d`

## Usage
Look into `understanding-clouds-kaggle/scripts` and change the parameters in the respective configs `understanding-clouds-kaggle/configs` if necessary. The current configs work for Google Colaboratory with non-mounted drives.

### Installation
Make sure that you already have `torch>=1.2.0` and `torchvision>=0.4.0` installed! Everything else can be easily installed through:
```
git clone https://github.com/jchen42703/understanding-clouds-kaggle.git
cd understanding-clouds-kaggle
pip install .
pip install git+https://github.com/qubvel/segmentation_models.pytorch
```
To install the most recent version of catalyst:
```
git clone https://github.com/catalyst-team/catalyst.git
cd catalyst
pip install .
```
Annnd you're basically done!

### Training
```
# classification
!python /content/understanding-clouds-kaggle/scripts/train_yaml.py --yml_path="/content/understanding-clouds-kaggle/configs/train_classification.yml"

# segmentation
!python /content/understanding-clouds-kaggle/scripts/train_yaml.py --yml_path="/content/understanding-clouds-kaggle/configs/train_seg.yml"
```

### Inference
```
# classification
!python understanding-clouds-kaggle/scripts/create_sub_no_trace_yaml.py --yml_path="/content/understanding-clouds-kaggle/configs/create_sub_clf.yml"

# segmentation
!python understanding-clouds-kaggle/scripts/create_sub_no_trace_yaml.py --yml_path="/content/understanding-clouds-kaggle/configs/create_sub_seg.yml"
```

### Cascade
```
from clouds.inference import combine_segmentation_classification_dfs
# removes FP with the classifier predictions
out_df = combine_segmentation_classification_dfs(seg_df, class_df)
# saves the cascaded results
out_df.to_csv("sub_final.csv", index=False)
```
