"""
Script generates predictions
"""
import argparse
from models.prepare_train_val import get_split
from models.dataset import SaltDataset
import cv2
from skimage import io, img_as_float
from models.models import AlbuNet
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import models.utils
#import models.prepare_data
from torch.utils.data import DataLoader
#from torch.nn import functional as F
#from prepare_data import (original_height,
#                          original_width,
#                          h_start, w_start
#                          )

#from models.transforms import (ImageOnly,
#                        AddMargin,
#                        Normalize,
#                        DualCompose)

from albumentations import  Compose, PadIfNeeded, Normalize, CenterCrop



import warnings
warnings.simplefilter("ignore", UserWarning)


def img_transform(p=1):
    return Compose([
        PadIfNeeded(min_height=128, min_width=128, border_mode=0, p=1),
        Normalize(mean=(0, 0, 0), std=(1, 1, 1), p=1)
    ], p=p)


original_height, original_width = 101, 101

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_model(model_path, model_type='AlbuNet', problem_type='binary'):
    """
    :param model_path:
    :param model_type: 'UNet', 'UNet16', 'UNet11', 'LinkNet34'
    :param problem_type: 'binary', 'parts', 'instruments'
    :return:
    """
    if problem_type == 'binary':
        num_classes = 1
    #elif problem_type == 'parts':
    #    num_classes = 4
    #elif problem_type == 'instruments':
    #    num_classes = 8

    if model_type == 'UNet16':
        model = UNet16(num_classes=num_classes)
    elif model_type == 'AlbuNet':
        model = AlbuNet(num_classes=num_classes)
    elif model_type == 'UNet11':
        model = UNet11(num_classes=num_classes)
    elif model_type == 'LinkNet34':
        model = LinkNet34(num_classes=num_classes)
    elif model_type == 'UNet':
        model = UNet(num_classes=num_classes)

    if torch.cuda.is_available():
        state = torch.load(str(model_path))
    else:
        state = torch.load(str(model_path), map_location = 'cpu')
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)

    if torch.cuda.is_available():
        return model.cuda()

    model.eval()

    return model


def predict(model, from_file_names, batch_size: int, to_path, problem_type):
    loader = DataLoader(
        dataset=SaltDataset(from_file_names, transform=img_transform(p=1), mode='predict'),
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )

    for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='Predict', ascii=True)):
        #inputs = utils.variable(inputs, volatile=True)

        inputs = inputs.to(device)
        outputs = model(inputs)

        for i, image_name in enumerate(paths):
            if problem_type == 'binary':
                #factor = prepare_data.binary_factor
                factor = 1.0
                t_mask = (torch.sigmoid(outputs[i, 0]).data.cpu().numpy() * factor).astype(float)


            #h, w = t_mask.shape
            #top = (h - original_height) // 2
            #bottom = top + original_height
            #left = (w - original_width) // 2
            #right = left + original_width
            #full_mask = t_mask[top:bottom, left:right]

            aug = CenterCrop(101, 101)
            augmented = aug(image=t_mask)
            full_mask = augmented["image"]


            out_folder = Path(paths[i]).parent.parent.name
            (to_path / out_folder).mkdir(exist_ok=True, parents=True)

            full_mask = img_as_float(full_mask)
            io.imsave(str(to_path / out_folder / (Path(paths[i]).stem + '.png')), full_mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_path', type=str, default='data/models/', help='path to model folder')
    arg('--model_type', type=str, default='AlbuNet', help='network architecture',
        choices=['AlbuNet','UNet', 'UNet11', 'UNet16', 'LinkNet34'])
    arg('--output_path', type=str, help='path to save images', default='data/predictions')
    arg('--batch-size', type=int, default=16)
    arg('--fold', type=int, default=0, choices=[0, 1, 2, 3, 4, -1], help='-1: all folds')
    arg('--problem_type', type=str, default='binary', choices=['binary', 'parts', 'instruments'])
    arg('--workers', type=int, default=6)

    args = parser.parse_args()

    if args.fold == -1:
        for fold in [0, 1, 2, 3, 4]:
            _, file_names = get_split(fold)
            model = get_model(str((Path(args.model_path) / args.model_type).joinpath('model_{fold}.pt'.format(fold=fold))),
                              model_type=args.model_type, problem_type=args.problem_type)

            print('num file_names = {}'.format(len(file_names)))

            output_path = Path(args.output_path) / args.model_type / 'OOF'
            output_path.mkdir(exist_ok=True, parents=True)

            predict(model, file_names, args.batch_size, output_path, problem_type=args.problem_type)
    else:
        _, file_names = get_split(args.fold)
        model = get_model(str((Path(args.model_path) / args.model_type).joinpath('model_{fold}.pt'.format(fold=args.fold))),
                          model_type=args.model_type, problem_type=args.problem_type)

        print('num file_names = {}'.format(len(file_names)))

        output_path = Path(args.output_path) / args.model_type / 'OOF'
        output_path.mkdir(exist_ok=True, parents=True)

        predict(model, file_names, args.batch_size, output_path, problem_type=args.problem_type)
