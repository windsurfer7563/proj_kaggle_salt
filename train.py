import argparse
import json
from pathlib import Path
from models.validation import validation_binary

import torch
from torch import nn

from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import torch.nn.functional as F

from models.models import AlbuNet, WindNet
from models.loss import dice_loss,jaccard

import models.lovasz_losses as LL
from models.dataset import SaltDataset
from models.utils import train

from models.prepare_train_val import get_split

import cv2
import numpy as np


from albumentations import (HorizontalFlip, VerticalFlip, Normalize,
    ShiftScaleRotate, Blur, OpticalDistortion,  GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur,
    MedianBlur, IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast, RandomBrightness,
    Flip, OneOf, Compose, RandomGamma, PadIfNeeded, CLAHE, InvertImg, ElasticTransform, IAAPerspective, RandomSizedCrop,
    RandomCrop, Resize, DualTransform)



def make_loader(file_names, args, config, shuffle=False, transform=None):
    return DataLoader(
        dataset=SaltDataset(file_names, transform=transform),
        shuffle=shuffle,
        num_workers=args.workers,
        batch_size= config['batch_size'],
        pin_memory=torch.cuda.is_available()
    )


def main():
    global config
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--loss', type=str, default = 'bce' ,choices = ['bce', 'lovash'])
    arg('--train_crop_height', type=int, default=128)
    arg('--train_crop_width', type=int, default=128)
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--fold', type=int, help='fold', default=0)
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--n-epochs', type=int, default=100)
    arg('--workers', type=int, default=4)
    arg('--model', type=str, default='WindNet', choices=['UNet', 'UNet11', 'AlbuNet'])
    arg('--freeze', type=int, default=0)
    arg('--warmup', type = int, default =0)


    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    num_classes = 1

    if args.model == 'UNet':
        model = UNet(num_classes=num_classes)
    elif args.model == 'UNet11':
        model = UNet11(num_classes=num_classes, pretrained='vgg')
    elif args.model == 'UNet16':
        model = UNet16(num_classes=num_classes, pretrained='vgg')
    elif args.model == 'AlbuNet':
        model = AlbuNet(num_classes=num_classes, pretrained=True)

    elif args.model == 'WindNet':
        model = WindNet(num_classes=num_classes, pretrained=True, dropout_2d=0)
        config = json.load(open('configs/WindNet.json'))

    else:
        model = UNet(num_classes=num_classes, input_channels=3)

    if args.freeze != 0:
        for param in model.encoder.parameters():
            param.requires_grad = False
    else:
        for param in model.encoder.parameters():
            param.requires_grad = True


    #loss = LL.binary_xloss
    #loss = LL.lovasz_hinge

    def lovash_loss(logit, truth):
        logit = logit.squeeze(1)
        truth = truth.squeeze(1)
        loss = 0.2 * LL.binary_xloss(logit, truth) + 0.8 * LL.lovasz_hinge(logit, truth, per_image=True)
        #loss = LL.lovasz_hinge(logit, truth, per_image=True)
        return loss

    def bcejaccdice_loss(output, target):
        bce = nn.BCEWithLogitsLoss()(output, target)
        output = torch.sigmoid(output)
        dice = dice_loss(output, target)
        jaccard_l = jaccard(output, target)

        loss = 0.4 * bce + 0.2 * (1 - dice) + 0.2 * (1 - jaccard_l)
        return loss

    criterion  = bcejaccdice_loss if args.loss == "bce" else lovash_loss


    cudnn.benchmark = True


    train_file_names, val_file_names = get_split(args.fold)

    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))


    def train_transform(p=1):
        return Compose([
            HorizontalFlip(p=0.5),
            OneOf([
                    RandomSizedCrop((91, 98), 101, 101, p=0.6),
                    ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit= 10, p=0.4),
            ], p=0.6),

            #OneOf([
            #    IAAAdditiveGaussianNoise(), #may by
            #    GaussNoise(),#may by
            #], p=0.2),
            #InvertImg(p = 0.2),
            #OneOf([
            #    MotionBlur(p=0.2),
            #    MedianBlur(blur_limit=3, p=0.3),
            #    Blur(blur_limit=3, p=0.5),
            #], p=0.4),
            OneOf([
                #ElasticTransform(p=.2), # bad
                #IAAPerspective(p=.2), #bad
                IAAPiecewiseAffine(p=.5),
                #OpticalDistortion(p=0.2),#bad
                GridDistortion(p=0.5),
            ], p=.2),
            OneOf([
                CLAHE(clip_limit=2),
                RandomGamma(),
                IAASharpen(),
                IAAEmboss(),
                RandomContrast(),
                RandomBrightness(),
            ], p=0.5),


        ], p=p)

    #def val_transform(p=1):
    #    return Compose([
    #       Normalize(mean=(0, 0, 0), std=(1, 1, 1), p=1)
    #    ], p=p)


    train_loader = make_loader(train_file_names, args, config, shuffle=True, transform=train_transform(p=0.9))
    valid_loader = make_loader(val_file_names, args, config, transform=None)

    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))

    valid = validation_binary

    train(
        args=args,
        model=model,
        config = config,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=valid,
        fold=args.fold,
        num_classes=num_classes
    )


if __name__ == '__main__':
    main()