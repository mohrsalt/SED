"""
Code from:
https://github.com/DCASE-REPO/DESED_task
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from sed_scores_eval.base_modules.scores import create_score_dataframe

from data_util.audioset_classes import indices
def batched_decode_preds(
        strong_preds,
        filenames,
        encoder,
        thresholds=[0.5],
        median_filter=None,
        pad_indx=None,
):
    """Decode a batch of predictions to dataframes. Each threshold gives a different dataframe and stored in a
    dictionary

    Args:
        strong_preds: torch.Tensor, batch of strong predictions.
        filenames: list, the list of filenames of the current batch.
        encoder: ManyHotEncoder object, object used to decode predictions.
        thresholds: list, the list of thresholds to be used for predictions.
        median_filter: int, the number of frames for which to apply median window (smoothing).
        pad_indx: list, the list of indexes which have been used for padding.

    Returns:
        dict of predictions, each keys is a threshold and the value is the DataFrame of predictions.
    """
    # Init a dataframe per threshold

    prediction_dfs = {}
    for threshold in thresholds:
        prediction_dfs[threshold] = pd.DataFrame()

    for j in range(len(strong_preds)):  # over batches
        audio_id = filenames[j].filename
        
        c_scores = strong_preds[j]
        c_scores = c_scores.squeeze(0)
        if pad_indx is not None:
            true_len = int(c_scores.shape[-1] * pad_indx[j].item())
            c_scores = c_scores[:true_len]
        c_scores = c_scores.transpose(0, 1).detach().cpu().numpy()

        if median_filter is not None:
            c_scores = scipy.ndimage.filters.median_filter(c_scores, (median_filter, 1))

        c_scores=c_scores[:,indices]

        for c_th in thresholds:
            pred = c_scores > c_th
            pred = encoder.decode_strong(pred, raw_labels=c_scores)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset","score"])
            pred["filename"] = audio_id
            prediction_dfs[c_th] = pd.concat(
                [prediction_dfs[c_th], pred], ignore_index=True
            )

    return prediction_dfs
