import numpy as np
import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def validation_binary(model: nn.Module, criterion, valid_loader, num_classes=None):
    model.eval()
    losses = []

    jaccard = []

    iou = []

    for inputs, targets in valid_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        jaccard += [get_jaccard(targets, (torch.sigmoid(outputs) > 0.5).float()).item()]
        iou += [get_iou((torch.sigmoid(outputs) > 0.5)).data.cpu().numpy().astype(np.uint8)]


    valid_loss = np.mean(losses).astype(float)  # type: float

    valid_jaccard = np.mean(jaccard).astype(float)

    valid_iou = np.mean(iou).astype(float)

    print('Valid loss: {:.5f}, jaccard: {:.5f}, mean iou: {:.5f}'.format(valid_loss, valid_jaccard, valid_iou))
    metrics = {'valid_loss': valid_loss, 'jaccard': valid_jaccard, 'mean_iou': valid_iou}
    return metrics


def get_jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1).sum(dim = -1)
    union = y_true.sum(dim=-2).sum(dim=-1).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1).sum(dim = -1)

    return (intersection / (union - intersection + epsilon)).mean()


def get_iou(A, B):
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

"""
def calculate_iou(confusion_matrix):
    ious = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = true_positives + false_positives + false_negatives
        if denom == 0:
            iou = 0
        else:
            iou = float(true_positives) / denom
        ious.append(iou)
    return ious


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
"""
