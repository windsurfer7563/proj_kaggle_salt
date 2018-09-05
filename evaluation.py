from pathlib import Path
import argparse
from skimage.io import imread
from skimage import img_as_float
import numpy as np
from tqdm import tqdm

"""
def general_dice(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for instrument_id in set(y_true.flatten()):
        if instrument_id == 0:
            continue
        result += [dice(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)


def general_jaccard(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for instrument_id in set(y_true.flatten()):
        if instrument_id == 0:
            continue
        result += [jaccard(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)
"""
def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)


# Precision helper function
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn

def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
            metric.append(1)
            continue

        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = np.sum(intersection > 0) / np.sum(union > 0)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--predictions_path', type=str, default='data/predictions/AlbuNet',
        help='path where predicted images are located')
    arg('--target_path', type=str, default='data/train/masks/', help='path with gt masks')
    #arg('--problem_type', type=str, default='parts', choices=['binary', 'parts', 'instruments'])
    args = parser.parse_args()

    result_dice = []
    result_jaccard = []
    pred_batch = []
    gt_batch = []


    for file_name in tqdm((Path(args.predictions_path).glob('*'))):
        y_pred = (img_as_float(imread(str(file_name))) > 0.4).astype(np.uint8)

        gt_file_name = Path(args.target_path) / file_name.name
        y_true = (imread(gt_file_name) > 0).astype(np.uint8)
        pred_batch.append(y_pred)
        gt_batch.append(y_true)

        result_dice += [dice(y_true, y_pred)]
        result_jaccard += [jaccard(y_true, y_pred)]



    print('Dice = ', np.mean(result_dice), np.std(result_dice))
    print('Jaccard = ', np.mean(result_jaccard), np.std(result_jaccard))
    pred_batch =  np.stack(pred_batch, axis=0)
    gt_batch = np.stack(gt_batch, axis=0)

    print('AP = ', get_iou_vector(gt_batch, pred_batch))