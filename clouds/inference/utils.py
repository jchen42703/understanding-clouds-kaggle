import numpy as np
import cv2
import torch

from functools import partial

def mask2rle(img):
    """
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def post_process(probability, threshold, min_size, pred_shape=(256, 400)):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    # don't remember where I saw it
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros(pred_shape, np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num

def get_df_histogram(df):
    """
    From: https://www.kaggle.com/lightforever/severstal-mlcomp-catalyst-infer-0-90672
    """
    df = df.fillna("")
    df["Image"] = df["ImageId_ClassId"].map(lambda x: x.split("_")[0])
    df["Class"] = df["ImageId_ClassId"].map(lambda x: x.split("_")[1])
    df["empty"] = df["EncodedPixels"].map(lambda x: not x)
    print(df[df["empty"] == False]["Class"].value_counts())

def load_weights_infer(checkpoint_path, model):
    """
    Loads pytorch model from a checkpoint and into inference mode.

    Args:
        checkpoint_path (str): path to a .pt or .pth checkpoint
        model (torch.nn.Module): <-
    Returns:
        Model with loaded weights and in evaluation mode
    """
    try:
        # catalyst weights
        state_dict = torch.load(checkpoint_path, map_location="cpu")["model_state_dict"]
    except:
        # anything else
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

def flip(x, dim):
    """
    From: https://github.com/MIC-DKFZ/nnUNet/blob/6150f1b0282daad11d4ad8d7227e68abe75e6f06/nnunet/utilities/tensor_utilities.py
    flips the tensor at dimension dim (mirroring!)

    Args:
        x: torch tensor
        dim: axis to flip across
    Returns:
        flipped torch tensor
    """
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def ud_flip(x):
    """
    Assumes the x has the shape: [batch, n_channels, h, w]
    """
    return flip(x, 2)

def lr_flip(x):
    """
    Assumes the x has the shape: [batch, n_channels, h, w]
    """
    return flip(x, 3)

def lrud_flip(x):
    """
    Assumes the x has the shape: [batch, n_channels, h, w]
    """
    return ud_flip(lr_flip(x))

def apply_nonlin(logit, non_lin="sigmoid"):
    """
    Applies non-linearity and sharpens if it's sigmoid.

    Args:
        logit (torch.Tensor): output logits from a model
            shape: (batch_size, n_classes, h, w) or (batch_size, n_classes)
        non_lin (str): one of [None, 'sigmoid', 'softmax']
    Returns:
        x: torch.Tensor, same shape as logit
    """
    if non_lin is None:
        return logit
    elif non_lin == "sigmoid":
        x = torch.sigmoid(logit)
        return x
    elif non_lin == "softmax":
        # softmax across the channels dim
        x = torch.softmax(logit, dim=1)
        return x

def tta_one(model, batch, mode, results_arr, num_results, post_process_fn, tta_fn, **tta_kwargs):
    """
    Args:
        model (nn.Module): model should be in evaluation mode
        batch (torch.Tensor): shape (batch_size, n_channels, h, w)
        mode (str): Either 'segmentation' or 'classification'
        results_arr (np.ndarray): contains the averaged tta results
            shape: (batch_size, n_channels, h, w)
        num_results (int): Number of tta ops + 1
        post_process_fn (function): takes in a torch.Tensor as input and
            processes the output, besides tta.
        tta_fn (function): The actual tta function; must have a torch.Tensor
            as the input and returns a torch.Tensor
        **tta_kwargs: For specifying specifc parameters for `tta_fn`
    Returns:
        returns results_arr with the new averaged in tta result
    """
    pred = model(tta_fn(batch, **tta_kwargs).cuda())
    if mode == "segmentation":
        results_arr += 1/num_results * post_process_fn(tta_fn(pred, **tta_kwargs))
    elif mode == "classification":
        results_arr += 1/num_results * post_process_fn(pred).squeeze()
    return results_arr

def tta_flips_fn(model, batch, mode="segmentation", flips=["lr_flip",],
                 non_lin="sigmoid"):
    """
    Inspired by: https://github.com/MIC-DKFZ/nnUNet/blob/2228cbe9e77910aaf97040790af83b8984ab9c11/nnunet/network_architecture/neural_network.py
    Applies flip TTA with cuda.

    Args:
        model (nn.Module): model should be in evaluation mode.
        batch (torch.Tensor): shape (batch_size, n_channels, h, w)
        mode (str): Either 'segmentation' or 'classification'
        flips (list-like): consisting one of or all of ["lr_flip", "ud_flip", "lrud_flip"].
            Defaults to ["lr_flip", "ud_flip", "lrud_flip"].
        non_lin (str): Either sigmoid or softmax
    Returns:
        averaged probability predictions
    """
    process = partial(apply_nonlin, non_lin=non_lin)
    with torch.no_grad():
        batch_size = batch.shape[0]
        spatial_dims = list(batch.shape[2:]) if mode=="segmentation" else []
        results = torch.zeros([batch_size, 4] + spatial_dims, dtype=torch.float).cuda()

        num_results = 1 + len(flips)
        pred = process(model(batch.cuda())).squeeze()
        results += 1/num_results * pred
        # applying tta
        tta_fn_list = [globals()[fn] for fn in flips]
        for tta_fn in tta_fn_list:
            results = tta_one(model, batch, mode, results, num_results,
                              post_process_fn=process,
                              tta_fn=tta_fn)
    return results
