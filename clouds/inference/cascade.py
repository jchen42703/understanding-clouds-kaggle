import numpy as np

def combine_segmentation_classification_dfs(df_segmentation, df_classification):
    """
    From: https://www.kaggle.com/bibek777/heng-s-model-inference-kernel
    Removes false positives from a segmentation model sub using classification model predictions.
    """
    df_mask = df_segmentation.fillna("").copy()
    df_label = df_classification.fillna("").copy()
    # do filtering using predictions from classification and segmentation models
    assert(np.all(df_mask["Image_Label"].values == df_label["Image_Label"].values))
    print((df_mask.loc[df_label["EncodedPixels"]=="","EncodedPixels"] != "").sum() ) #202
    df_mask.loc[df_label["EncodedPixels"]=="","EncodedPixels"]=""
    return df_mask
