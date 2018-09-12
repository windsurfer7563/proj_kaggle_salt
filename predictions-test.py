"""
Script generates predictions
"""
import os
import argparse
from models.dataset import data_path
from models.dataset import SaltDataset
from skimage import io, img_as_float
from models.models import AlbuNet, WindNet
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader

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


def get_models(models_path, model_type='AlbuNet'):
    """
    :param models_path:
    :param model_type: 'UNet', 'UNet16', 'UNet11', 'LinkNet34'
    :return:
    """
    num_classes = 1
    models = []

    for fold in [0, 1, 2, 3, 4]:

        model_path = str(models_path.joinpath('model_{fold}.pt'.format(fold=fold)))

        if model_type == 'UNet16':
            model = UNet16(num_classes=num_classes)
        elif model_type == 'AlbuNet':
            model = AlbuNet(num_classes=num_classes)
        elif model_type == 'WindNet':
            model = WindNet(num_classes=num_classes)
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
            model.cuda()

        model.eval()

        models.append(model)

    return models


def predict(models, from_file_names, batch_size: int, to_path):
    # TODO Zero out masks with less then 30 pixels in mask
    loader = DataLoader(
        dataset=SaltDataset(from_file_names, transform=img_transform(p=1), mode='test'),
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )
    print("Start predictions....")
    print(len(models))
    with torch.no_grad():
        for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='Predict', ascii=True)):

            outputs_arr = []
            inputs = inputs.to(device)

            for model in models:
                outputs_arr.append(torch.sigmoid(model(inputs)).data.cpu().numpy().astype(float))

            outputs_arr = np.stack(outputs_arr, axis=0)

            outputs_arr = outputs_arr.sum(axis=0)/len(models) # now arr contains one mean mask for each image in batch

            for i, image_name in enumerate(paths):

                t_mask = outputs_arr[i, 0]

                top = 13
                left = 13
                bottom = top + 101
                right = left + 101
                full_mask = t_mask[top:bottom, left:right]

                #aug = CenterCrop(101, 101)
                #augmented = aug(image=t_mask)
                #full_mask = augmented["image"]

                full_mask = img_as_float(full_mask)

                out_folder = Path(paths[i]).parent.parent.name
                (Path(to_path) / out_folder).mkdir(exist_ok=True, parents=True)

                io.imsave(str(to_path / out_folder / (Path(paths[i]).stem + '.png')), full_mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_path', type=str, default='data/models/', help='path to model folder')
    arg('--model_type', type=str, default='WindNet', help='network architecture',
        choices=['AlbuNet','WindNet','UNet', 'UNet11', 'UNet16', 'LinkNet34'])
    arg('--output_path', type=str, help='path to save images', default='data/predictions/')
    arg('--batch-size', type=int, default=64)
    arg('--problem_type', type=str, default='binary', choices=['binary', 'parts', 'instruments'])
    arg('--workers', type=int, default=6)

    args = parser.parse_args()

    models = get_models(Path(args.model_path) / args.model_type, model_type=args.model_type)

    test_path = os.path.join(data_path, 'test', 'images')
    ids = next(os.walk(test_path))[2]

    file_names = [i[:-4] for i in ids]

    print('num file_names = {}'.format(len(file_names)))

    output_path = Path(args.output_path) / args.model_type / 'test'
    output_path.mkdir(exist_ok=True, parents=True)

    predict(models, file_names, args.batch_size, output_path)