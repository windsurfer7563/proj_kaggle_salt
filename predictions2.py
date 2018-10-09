"""
Script generates predictions
"""
import argparse
from models.prepare_train_val import get_split
from models.dataset import SaltDataset
import cv2
from skimage import io, img_as_float
from models.models import AlbuNet, ResNet34, SE_ResNext50,  SE_ResNext50_2
from models.inceptionv3.unet import Incv3
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
from collections import namedtuple
import shutil
import pandas as pd


from torch.utils.data import DataLoader

#from albumentations import  Compose, PadIfNeeded, Normalize, CenterCrop

import warnings
warnings.simplefilter("ignore", UserWarning)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_model(model_path, model_type='SE_ResNext50_2'):
    """
    :param model_path:
    :param model_type: 'UNet', 'UNet16', 'UNet11', 'LinkNet34'
    :return:
    """

    num_classes = 1

    if model_type == 'ResNet34':
        model = ResNet34(num_classes=num_classes)
    elif model_type == 'SE_ResNext50_2':
        model = SE_ResNext50_2(num_classes=num_classes)
    else:
        print("model type not defined")
        raise NotImplementedError

    print('Model_path: ', model_path)

    if torch.cuda.is_available():
        state = torch.load(model_path)
    else:
        state = torch.load(model_path, map_location='cpu')

    state = {key.replace('module.', '', 1): value for key, value in state['model'].items()}

    model.load_state_dict(state)

    if torch.cuda.is_available():
        return model.cuda()

    return model


def predict(model, config, from_file_names, batch_size: int, to_path):
    loader = DataLoader(
        dataset=SaltDataset(from_file_names, config, transform=None, mode='predict'),
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )
    model.eval()
    with torch.no_grad():
        outputs_image = {}

        for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='Predict', ascii=True)):
            inputs = inputs.to(device)
            outputs, logit_image = model(inputs)

            for i, image_name in enumerate(paths):
                t_mask = (torch.sigmoid(outputs[i, 0]).data.cpu().numpy()).astype(float)
                outputs_image[image_name] = torch.sigmoid(logit_image[i]).item()

                top = 13
                left = 13
                bottom = top + 101
                right = left + 101
                full_mask = t_mask[top:bottom, left:right]

                to_path.mkdir(exist_ok=True, parents=True)

                np.save(str(to_path / (Path(paths[i]).stem + '.npy')), full_mask)

            outputs_image = pd.DataFrame.from_dict(outputs_image, orient='index')
            outputs_image.to_csv(str(to_path / 'images.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_path', type=str, default='data/models/', help='path to model folder')
    arg('--model_type', type=str, default='SE_ResNext50_2', help='network architecture',
        choices=['ResNet34', 'IncV3', 'SE_ResNext50_2'])
    arg('--output_path', type=str, help='path to save images', default='data/predictions')
    arg('--batch-size', type=int, default=32)
    arg('--fold', type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7,  -1], help='-1: all folds')
    arg('--workers', type=int, default=4)
    arg('--config', default='SE_ResNext50_finetune.json')

    args = parser.parse_args()

    cfg = json.load(open('configs/' + args.config))
    Config = namedtuple("Config", cfg.keys())
    config = Config(**cfg)

    if args.fold == -1:
        for fold in [0, 1, 2, 3, 4, 5 ,6 ,7]:
            _, file_names = get_split(fold)
            model_path = str((Path(args.model_path) / args.model_type).joinpath('model_{fold}.pt'.format(fold=fold)))
            print("start model load.")
            model = get_model(model_path,
                              model_type=args.model_type)
            print("model loaded.")
            print('num file_names = {}'.format(len(file_names)))

            output_path = Path(args.output_path) / args.model_type / 'OOF'
            output_path.mkdir(exist_ok=True, parents=True)

            predict(model, config, file_names, args.batch_size, output_path)
    else:
        print("Predictions for fold: {}".format(args.fold))
        _, file_names = get_split(args.fold)
        print(len(file_names))
        model_path = str((Path(args.model_path) / args.model_type).joinpath('model_{fold}.pt'.format(fold=args.fold)))
        model = get_model(model_path,
                          model_type=args.model_type)

        print('num file_names = {}'.format(len(file_names)))

        output_path = Path(args.output_path) / args.model_type / 'OOF'
        shutil.rmtree(str(output_path),ignore_errors=True)
        output_path.mkdir(exist_ok=True, parents=True)



        predict(model, config, file_names, args.batch_size, output_path)
