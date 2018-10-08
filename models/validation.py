import numpy as np
import torch
from torch import nn
import gc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def validation_binary(model: nn.Module, criterion, valid_loader, num_classes=None):
    model.eval()
    losses = []
    acc = []
    jaccard = []

    iou = []
    pred_batch = []
    pred_batch_images = []
    gt_batch = []

    model.eval()
    with torch.no_grad():
        for inputs, targets in valid_loader:
            batch_size = inputs.size()[0]

            targets_cuda = targets.to(device)
            outputs, logit_image = model(inputs)
            loss_pixel, loss_image = criterion(outputs, logit_image, targets_cuda)
            loss = loss_pixel + loss_image
            losses.append(loss.item())

            # get original img size for precision metric calculation
            top = 13
            left = 13
            bottom = top + 101
            right = left + 101
            targets = targets[:, :, top:bottom, left:right]
            targets_cuda = targets_cuda[:, :, top:bottom, left:right]
            outputs = outputs[:, :, top:bottom, left:right]

            outputs_bin = (torch.sigmoid(outputs) > 0.42).to(torch.uint8)

            outputs_image_bin = (torch.sigmoid(logit_image) > 0.4).to(torch.uint8)


            jaccard += [get_jaccard(targets_cuda, outputs_bin.float()).item()]

            outputs_bin = outputs_bin.cpu().numpy().astype(np.uint8)
            pred_batch.append(outputs_bin.squeeze(1))
            outputs_image_bin = outputs_image_bin.cpu().numpy().astype(np.uint8)
            pred_batch_images.append(outputs_image_bin)
            gt_batch.append(targets.squeeze(1))



    valid_loss = np.mean(losses).astype(float)
    valid_jaccard = np.mean(jaccard).astype(float)
    #valid_iou = np.mean(iou).astype(float)

    pred_batch = np.concatenate(pred_batch, axis=0)
    pred_batch_images = np.concatenate(pred_batch_images, axis=0)
    gt_batch = np.concatenate(gt_batch, axis=0)

    pred_batch = pred_batch * pred_batch_images[:, np.newaxis, np.newaxis]

    iou2 = get_iou2(gt_batch, pred_batch)

    targets_images = (gt_batch.sum(axis=(1, 2)) > 0)
    a = (pred_batch_images == targets_images).sum().item()/ pred_batch_images.shape[0]
    acc.append(a)

    acc = np.mean(acc)

    print('Valid loss: {:.5f}, jaccard: {:.5f}, mean iou2: {:.5f}, acc: {:.5f}'.format(valid_loss, valid_jaccard, iou2, acc))
    metrics = {'valid_loss': valid_loss, 'jaccard': valid_jaccard, 'mean_iou2': iou2, 'acc': acc}
    return metrics


def get_jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1).sum(dim = -1)
    union = y_true.sum(dim=-2).sum(dim=-1).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1).sum(dim = -1)

    return (intersection / (union - intersection + epsilon)).mean()
"""
There are 2 metrics calculation example on Kaggle forum (for pytorch).
Found 2nd one more stable
"""




def get_iou(labels: torch.Tensor, outputs: torch.Tensor):
        SMOOTH = 1e-6
        # You can comment out this line if you are passing tensors of equal shape
        # But if you are passing output from UNet or something it will most probably
        # be with the BATCH x 1 x H x W shape
        outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
        labels = labels.squeeze(1)
        intersection = (outputs & labels).sum((1, 2)).float()  # Will be zero if Truth=0 or Prediction=0
        union = (outputs | labels).sum((1, 2)).float()  # Will be zzero if both are 0

        iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

        thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

        return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch

def get_iou2(A, B):
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



def calculate_dice(confusion_matrix):
    dices = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            dice = 0
        else:
            dice = 2 * float(true_positives) / denom
        dices.append(dice)
    return dices

