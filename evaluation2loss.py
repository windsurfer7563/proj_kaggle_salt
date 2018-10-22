from pathlib import Path
import argparse
from skimage.io import imread
from skimage import img_as_float
import numpy as np
from tqdm import tqdm
import pandas as pd
import cv2
import pickle

from skimage.morphology import remove_small_holes, remove_small_objects, binary_erosion, binary_dilation, disk, \
        closing, opening, dilation

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

#
# Below there is 2 metric realization
#

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

    return np.mean(metric), np.std(metric)


def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    # Jiaxin fin that if all zeros, then, the background is treated as object
    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0, 0.5, 1], [0, 0.5, 1]))
    #     temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))
    # print(temp1)
    intersection = temp1[0]
    # print("temp2 = ",temp1[1])
    # print(intersection.shape)
    # print(intersection)
    # Compute areas (needed for finding the union between all objects)
    # print(np.histogram(labels, bins = true_objects))
    area_true = np.histogram(labels, bins=[0, 0.5, 1])[0]
    # print("area_true = ",area_true)
    area_pred = np.histogram(y_pred, bins=[0, 0.5, 1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    intersection[intersection == 0] = 1e-9

    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def iou_metric_batch(y_true_in, y_pred_in):
    #y_pred_in = y_pred_in > 0.5  # added by sgx 20180728
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    # print("metric = ",metric)
    return np.mean(metric)


def get_acc(gt, pred, img_threshold):
    pred = (pred[:, 0, 0] > img_threshold).astype(np.uint8)
    #print(pred.shape)
    targets_images = (gt.sum(axis=(1, 2)) > 0).astype(np.uint8)
    #print(targets_images.shape)
    acc = (pred == targets_images).astype(np.uint8).sum().item() / pred.shape[0]
    return acc


if __name__ == '__main__':
    neib_df = pickle.load(open('neib_df2.pkl','rb'))
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--predictions_path', type=str, default='data/predictions/SE_ResNext50_2/OOF',
        help='path where predicted images are located')
    arg('--target_path', type=str, default='data/train/masks/', help='path with gt masks')
    #arg('--problem_type', type=str, default='parts', choices=['binary', 'parts', 'instruments'])
    args = parser.parse_args()

    result_dice = []
    result_jaccard = []
    pred_batch = []
    images_batch = []
    gt_batch = []

    threshold = 0.42
    img_threshold = 0.478
    images = pd.read_csv(str(Path(args.predictions_path) / 'images.csv'), index_col =0, header = None)
    images.columns = ['p']
    corr_no = 0
    for file_name in tqdm(Path(args.predictions_path).glob('*.npy'), ascii=True):

        y_pred = np.load(str(file_name))
        #print(y_pred)
        img_id = Path(file_name.stem)

        gt_file_name = Path(args.target_path) / Path(file_name.stem).with_suffix('.png')
        y_true = (imread(gt_file_name) > 0).astype(np.uint8)
        pred_batch.append(y_pred)
        gt_batch.append(y_true)
        p = images.loc[str(img_id)+'.png', 'p']

        id_ = str(img_id)
        if id_ in neib_df.index:
            corr = neib_df.loc[id_,'max_corr']
            if corr > 0.03:
                p = 1
                corr_no += 1
            else:
                zero_corr_no = (neib_df.loc[id_,['left_corr','right_corr', 'top_corr','bottom_corr']] == 0).sum()
                if zero_corr_no == 4:
                    p = p * 0.5


        img = np.ones((101,101)) * p
        images_batch.append(img)

        #result_dice += [dice(y_true, (y_pred > threshold)*(img > img_threshold))]
        #result_jaccard += [jaccard(y_true, (y_pred > threshold)*(img > img_threshold).astype(np.uint8))]

    print(corr_no)

    #print('Dice = ', np.mean(result_dice), np.std(result_dice))
    #print('Jaccard = ', np.mean(result_jaccard), np.std(result_jaccard))

    pred_batch = np.stack(pred_batch, axis=0)
    gt_batch = np.stack(gt_batch, axis=0)
    #print(pred_batch.shape)
    images_batch = np.stack(images_batch, axis=0)
    #print(images_batch.shape)

    #pred_batch[images_batch > img_threshold] = 0

    preds = ((pred_batch > threshold)*(images_batch > img_threshold)).astype(np.uint8)

    for i in range(preds.shape[0]):
        if preds[i,:,:].mean()==1:
            preds[i,:,:] = 0

        #if any(preds[i,0,:]) == 0 and any(preds[i,-1,:]) == 0 and any(preds[i,:,0]) == 0 and any(preds[i,:,-1]) == 0:
        #    preds[i,:,:] = 0
        #    print("ssss")


    m,s = get_iou_vector(gt_batch, preds)
    print('AP = {}, std = {}'.format(m,s))
    #print('AP2 = {}'.format(iou_metric_batch(gt_batch, ((pred_batch > threshold)*(images_batch > img_threshold)).astype(np.uint8))))



    #print("Finding best threshold to binarize...")
    #thresholds = np.linspace(0.3, 0.7, 31)
    #ious = np.array(
    #    [get_iou_vector(gt_batch, ((pred_batch>threshold)*(images_batch > img_threshold)).astype(np.uint8))
    #     for threshold in tqdm(thresholds, ascii=True)])

    #threshold_best_index = np.argmax(ious[:,0])
    #iou_best = ious[threshold_best_index]
    #threshold_best = thresholds[threshold_best_index]
    #print("Best IOU: {} at {}".format(iou_best, threshold_best))


    #print("Finding best threshold to binarize 2...")

    #thresholds = np.linspace(0.3, 0.7, 31)
    #ious = np.array(
    #    [get_iou_vector(gt_batch, ((pred_batch > threshold) * (images_batch > img_threshold)).astype(np.uint8)) for
    #          threshold in tqdm(thresholds, ascii=True)])


    #threshold_best_index = np.argmax(ious[:, 0])
    #iou_best = ious[threshold_best_index]
    #threshold_best = thresholds[threshold_best_index]
    #print("Best IOU: {} at {}".format(iou_best, threshold_best))

    img_tr =  np.linspace(0.38, 0.51, 13)
    #pix_tr = np.linspace(0.38, 0.5, 20)
    pix_tr = [0.42]

    best_iou = 0
    best_i_t = 0
    best_p_t = 0
    #for i_t in img_tr:
    #   for p_t in pix_tr:
    #       iou, _ = get_iou_vector(gt_batch, ((pred_batch > p_t) * (images_batch > i_t)).astype(np.uint8))
    #       if iou > best_iou:
    #            best_iou = iou
    #            best_i_t = i_t
    #            best_p_t = p_t

    #print("Best IOU: {} at p_t: {} i_t: {} ".format(best_iou, best_p_t, best_i_t))

