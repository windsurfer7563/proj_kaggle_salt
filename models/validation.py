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
        targets = targets.to(device)
        outputs = model(inputs.to(device))
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        jaccard += [get_jaccard(targets, (torch.sigmoid(outputs) > 0.45).float()).item()]
        iou += [get_iou(targets.to(torch.uint8), (torch.sigmoid(outputs) > 0.45).to(torch.uint8))]


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

