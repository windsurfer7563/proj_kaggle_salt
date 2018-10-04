"""
Script generates predictions
"""
import argparse
from models.prepare_train_val import get_split
from models.dataset import SaltDataset
import cv2
from skimage import io, img_as_float
from models.models import AlbuNet, ResNet34, SE_ResNext50
from models.inceptionv3.unet import Incv3
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader

#from albumentations import  Compose, PadIfNeeded, Normalize, CenterCrop

import warnings
warnings.simplefilter("ignore", UserWarning)


#def img_transform(p=1):
#    return Compose([
#        #PadIfNeeded(min_height=128, min_width=128, border_mode=0, p=1),
#        #Normalize(mean=(0, 0, 0), std=(1, 1, 1), p=1)
#    ], p=p)


#original_height, original_width = 101, 101

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_model(model_path, model_type='SE_ResNext50'):
    """
    :param model_path:
    :param model_type: 'UNet', 'UNet16', 'UNet11', 'LinkNet34'
    :return:
    """

    num_classes = 1

    if model_type == 'ResNet34':
        model = ResNet34(num_classes=num_classes)
    elif model_type == 'SE_ResNext50':
        model = SE_ResNext50(num_classes=num_classes)
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


def predict(model, from_file_names, batch_size: int, to_path, problem_type):
    loader = DataLoader(
        dataset=SaltDataset(from_file_names, transform=None, mode='predict', resize_x2=False),
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )
    model.eval()
    for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='Predict', ascii=True)):
        inputs = inputs.to(device)
        outputs = model(inputs)

        for i, image_name in enumerate(paths):
            t_mask = (torch.sigmoid(outputs[i, 0]).data.cpu().numpy()).astype(float)

            top = 13
            left = 13
            bottom = top + 101
            right = left + 101
            full_mask = t_mask[top:bottom, left:right]

            to_path.mkdir(exist_ok=True, parents=True)

            np.save(str(to_path / (Path(paths[i]).stem + '.npy')), full_mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_path', type=str, default='data/models/', help='path to model folder')
    arg('--model_type', type=str, default='SE_ResNext50', help='network architecture',
        choices=['ResNet34', 'IncV3', 'SE_ResNext50'])
    arg('--output_path', type=str, help='path to save images', default='data/predictions')
    arg('--batch-size', type=int, default=16)
    arg('--fold', type=int, default=0, choices=[0, 1, 2, 3, 4, -1], help='-1: all folds')
    arg('--workers', type=int, default=4)

    args = parser.parse_args()

    if args.fold == -1:
        for fold in [0, 1, 2, 3, 4]:
            _, file_names = get_split(fold)
            model_path = str((Path(args.model_path) / args.model_type).joinpath('model_{fold}.pt'.format(fold=fold)))
            print("start model load.")
            model = get_model(model_path,
                              model_type=args.model_type)
            print("model loaded.")
            print('num file_names = {}'.format(len(file_names)))

            output_path = Path(args.output_path) / args.model_type / 'OOF'
            output_path.mkdir(exist_ok=True, parents=True)

            predict(model, file_names, args.batch_size, output_path)
    else:
        _, file_names = get_split(args.fold)
        model_path = str((Path(args.model_path) / args.model_type).joinpath('model_{fold}.pt'.format(fold=args.fold)))
        model = get_model(model_path,
                          model_type=args.model_type)

        print('num file_names = {}'.format(len(file_names)))

        output_path = Path(args.output_path) / args.model_type / 'OOF'
        output_path.mkdir(exist_ok=True, parents=True)

        predict(model, file_names, args.batch_size, output_path)
