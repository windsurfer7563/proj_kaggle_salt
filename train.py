import argparse
import json
from pathlib import Path
from models.validation import validation_binary
import random
import torch
from torch import nn

from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from models.models import AlbuNet, ResNet34, SE_ResNext50, SE_ResNext50_2, SE_ResNext101
from models.inceptionv3.unet import Incv3
from models.loss import dice_loss,jaccard

import models.lovasz_losses as LL
from models.dataset import SaltDataset
from models.utils import train
from models.prepare_train_val import get_split

from collections import namedtuple

from albumentations.augmentations import functional as F

from albumentations import (HorizontalFlip,
    ShiftScaleRotate, Blur, OpticalDistortion,  GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur,
    MedianBlur, IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast, RandomBrightness,
    Flip, OneOf, Compose, RandomGamma, PadIfNeeded, CLAHE, InvertImg, ElasticTransform, IAAPerspective, RandomSizedCrop,
    RandomCrop, Resize, DualTransform, ImageOnlyTransform, RGBShift)


def make_loader(file_names, args, config, shuffle=False, transform=None):
    return DataLoader(
        dataset=SaltDataset(file_names, config, transform=transform),
        shuffle=shuffle,
        num_workers=args.workers,
        batch_size=config.batch_size,
        pin_memory=torch.cuda.is_available()
    )


def main():
    global config
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--n-epochs', type=int, default=100)
    arg('--fold', type=int, default=0)
    arg('--workers', type=int, default=4)
    arg('--freeze', type=int, default=0)
    arg('--warmup', type = int, default =0)
    arg('--config', type=str, default='ResNet34.json')
    arg('--freeze-bn', type=int, default=0)
    arg('--resume', type = int, default=0)
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')

    # arg('--epoch', type=int, default=0)
    # arg('--fold', type=int, help='fold', default=0)

    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    num_classes = 1


    cfg = json.load(open('configs/' + args.config))

    # just convenient way to access config items by config.item in opposite to config['item']
    Config = namedtuple("Config", cfg.keys())
    config = Config(**cfg)

    if config.model == 'ResNet34':
        model = ResNet34(num_classes=num_classes, pretrained=True, dropout_2d=0)
    elif config.model == 'SE_ResNext50':
        model = SE_ResNext50(num_classes=num_classes)

    elif config.model == 'SE_ResNext50_2':
        model = SE_ResNext50_2(num_classes=num_classes)

    elif config.model == 'SE_ResNext101':
        model = SE_ResNext101(num_classes=num_classes)
    elif config.model == 'IncV3':
        model = Incv3(num_classes=num_classes)
    else:
        raise NotImplementedError

    # encoder part could be freeze at the initial stage of training
    if args.freeze != 0:
        for param in model.encoder.parameters():
            param.requires_grad = False
    else:
        for param in model.encoder.parameters():
            param.requires_grad = True

    def lovash_loss(logit, truth):
        bce = nn.BCEWithLogitsLoss()(logit, truth)
        logit = logit.squeeze(1)
        truth = truth.squeeze(1)

        loss = 0.1 * bce + 0.9 * LL.lovasz_hinge(logit, truth, per_image=True)
        #loss = 0.1 * LL.binary_xloss(logit, truth) + 0.9 * LL.lovasz_hinge(logit, truth, per_image=True)
        #loss = LL.lovasz_hinge(logit, truth, per_image=True)
        return loss

    #def lovash_loss2(logit, logit_pixel, logit_image, truth):
    def lovash_loss2(logit_pixel, logit_image, truth):
        #top = 13
        #left = 13
        #bottom = top + 101
        #right = left + 101
        #logit_pixel = logit_pixel[:, :, top:bottom, left:right]
        #truth = truth[:, :, top:bottom, left:right]


        truth_image = (truth.sum(dim=[2, 3]) > 0).to(torch.float).view(-1)
        loss_image = nn.BCEWithLogitsLoss()(logit_image, truth_image)

        logit_pixel_non_zero = logit_pixel[(truth_image > 0)]
        truth_non_zero = truth[(truth_image > 0)]

        if list(logit_pixel_non_zero.size())[0] != 0:
            loss_pixel = lovash_loss(logit_pixel_non_zero, truth_non_zero)
        else:
            loss_pixel = 0

        #weight, weight_pixel, weight_image,  = 1, 0.5, 0.05
        weight_pixel, weight_image = 2, 0.08

        return weight_pixel * loss_pixel, weight_image * loss_image



    def bcejaccdice_loss(output, target):
        bce = nn.BCEWithLogitsLoss()(output, target)
        output = torch.sigmoid(output)
        dice = dice_loss(output, target)
        #jaccard_l = jaccard(output, target)
        #loss = 0.4 * bce + 0.2 * (1 - dice) + 0.2 * (1 - jaccard_l)
        loss = 0.5 * bce + 0.5 * (1 - dice)
        return loss

    criteries = {"bce": bcejaccdice_loss, "lovash": lovash_loss,"lovash2": lovash_loss2}
    criterion = criteries[config.loss]

    train_file_names, val_file_names = get_split(args.fold)
    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))

    def train_transform(p=1):
        return Compose([
            HorizontalFlip(p=0.5),
            OneOf([
                    RandomSizedCrop((92, 98), 101, 101,  p=0.6),
                    ShiftScaleRotate(shift_limit=(0, 0.1), scale_limit=(0, 0.05), rotate_limit=10, p=0.4),
            ], p=0.6),
            #OneOf([
            #    IAAAdditiveGaussianNoise(), #may by
            #    GaussNoise(),#may by
            #], p=0.2),
            #OneOf([
            #    MotionBlur(p=0.2),
            #    MedianBlur(blur_limit=3, p=0.3),
            #    Blur(blur_limit=3, p=0.5),
            #], p=0.4),
            OneOf([
                ElasticTransform(p=0.2, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                IAAPiecewiseAffine(p=.4),
                GridDistortion(p=0.4),
            ], p=.4),
            OneOf([
                #CLAHE(clip_limit=2),
                RandomGamma((90,110)),
                ShiftBrightness((5, 20)),
                IAAEmboss((0.1, 0.4), (0.1, 0.6)),
                RandomContrast(0.08),
                RandomBrightness(0.08),
            ], p=0.5),
        ], p=p)

    class ShiftBrightness(ImageOnlyTransform):
        def __init__(self, shift_limit=(5, 20), p=0.5):
            super(ShiftBrightness, self).__init__(p)
            self.limit = shift_limit

        def apply(self, image, shift=0, **params):
            return F.shift_rgb(image, shift, shift, shift)

        def get_params(self):
            return {'shift': random.uniform(self.limit[0], self.limit[1])}


    def train_transform_for_rn101(p=1):
        return Compose([
            HorizontalFlip(p=0.5),
            OneOf([
                    RandomSizedCrop((92, 98), 101, 101,  p=0.6),
                    ShiftScaleRotate(shift_limit=(0, 0.1), scale_limit=(0.05, 0.05), rotate_limit=8, p=0.4),
            ], p=0.8),
            OneOf([
                ElasticTransform(p=0.4, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                IAAPiecewiseAffine(p=.4),
                GridDistortion(p=0.5),
            ], p=.4),
            OneOf([
                RandomGamma((90, 110)),
                ShiftBrightness((5, 20)),
                RandomContrast(0.08),
                RandomBrightness(0.1),
            ], p=1),
        ], p=p)

    if config.model == 'SE_ResNext101':
        transforms = train_transform_for_rn101(p=1)
    else:
        transforms = train_transform(p=1)
             


    train_loader = make_loader(train_file_names, args, config, shuffle=True, transform=transforms)
    valid_loader = make_loader(val_file_names, args, config, transform=None)

    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))

    valid = validation_binary

    cudnn.benchmark = True
    train(
        args=args,
        model=model,
        config=config,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=valid,
        fold=args.fold,
        num_classes=num_classes
    )


if __name__ == '__main__':
    main()