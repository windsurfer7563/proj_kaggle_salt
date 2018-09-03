import argparse
import json
from pathlib import Path
from models.validation import validation_binary

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.backends.cudnn

from models.models import AlbuNet
from models.loss import LossBinary
from models.dataset import SaltDataset
from models.utils import train

from models.prepare_train_val import get_split

from models.transforms import (DualCompose,
                        ImageOnly,
                        Normalize,
                        HorizontalFlip,
                        VerticalFlip,
                        Rotate,
                        RandomBrightness,
                        RandomContrast,
                        AddMargin
                       )


def make_loader(file_names, args, shuffle=False, transform=None):
    return DataLoader(
        dataset=SaltDataset(file_names, transform=transform),
        shuffle=shuffle,
        num_workers=args.workers,
        batch_size=args.batch_size,
        pin_memory=torch.cuda.is_available()
    )


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--jaccard-weight', default=1, type=float)
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--fold', type=int, help='fold', default=0)
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--batch-size', type=int, default=1)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=2)
    arg('--model', type=str, default='AlbuNet', choices=['UNet', 'UNet11', 'AlbuNet'])
    arg('--freeze', type=int, default=0)
    #arg('--type', type=str, default='binary', choices=['binary', 'parts', 'instruments'])


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
    else:
        model = UNet(num_classes=num_classes, input_channels=3)

    if args.freeze != 0:
        for param in model.encoder.parameters():
            param.requires_grad = False
    else:
        for param in model.encoder.parameters():
            param.requires_grad = True

    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()

    loss = LossBinary(jaccard_weight=args.jaccard_weight)

    #if args.type == 'binary':
    #    loss = LossBinary(jaccard_weight=args.jaccard_weight)
    #else:
    #    loss = LossMulti(num_classes=num_classes, jaccard_weight=args.jaccard_weight)

    cudnn.benchmark = True


    train_file_names, val_file_names = get_split(args.fold)

    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))

    train_transform = DualCompose([
        AddMargin(128),
        HorizontalFlip(),
        VerticalFlip(),
        ImageOnly(RandomBrightness()),
        ImageOnly(RandomContrast()),
        ImageOnly(Normalize())
    ])

    val_transform = DualCompose([
        AddMargin(128),
        ImageOnly(Normalize())
    ])

    train_loader = make_loader(train_file_names, args, shuffle=True, transform=train_transform)
    valid_loader = make_loader(val_file_names, args, transform=val_transform)

    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))

    #if args.type == 'binary':
    #    valid = validation_binary
    #else:
    #    valid = validation_multi
    valid = validation_binary

    train(
        init_optimizer=lambda lr: Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr),
        args=args,
        model=model,
        criterion=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=valid,
        fold=args.fold,
        num_classes=num_classes
    )


if __name__ == '__main__':
    main()