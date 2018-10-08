"""
Script generates predictions
"""
import os
import argparse
from models.dataset import data_path
from models.dataset import SaltDataset
from models.models import ResNet34, SE_ResNext50
import torch
from pathlib import Path
import tqdm
import numpy as np
import json
from collections import namedtuple

from torch.utils.data import DataLoader

import warnings
warnings.simplefilter("ignore", UserWarning)

original_height, original_width = 101, 101

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_model_and_paths(models_path, model_type='SE_ResNext50', fold_no = 0 ):
    """
    :param models_path:
    :param model_type: 'UNet', 'UNet16', 'UNet11', 'LinkNet34'
    :return:
    """
    num_classes = 1

    if model_type == 'ResNet34':
        model = ResNet34(num_classes=num_classes)
    elif model_type == 'SE_ResNext50':
        model = SE_ResNext50(num_classes=num_classes)
    else:
        raise NotImplementedError

    paths = [model_path for model_path in models_path.glob('model_best_{}_*.pt'.format(fold_no))]

    return model, paths


def predict(model, config, model_paths,  from_file_names, batch_size, to_path, tta=1, fold_no=0):

    loader = DataLoader(
        dataset=SaltDataset(from_file_names, config, transform=None, mode='test'),
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )

    print("Start predictions...., TTA = {}".format(tta))

    fold_predictions = []

    for model_path in model_paths:
        if torch.cuda.is_available():
            state = torch.load(str(model_path))
        else:
            state = torch.load(str(model_path), map_location='cpu')
        state = {key.replace('module.', '', 1): value for key, value in state['model'].items()}
        model.load_state_dict(state)

        if torch.cuda.is_available():
            model.cuda()
        model.eval()

        all_predictions = []
        tq = tqdm.tqdm(total=(len(loader) * batch_size), ascii=True)
        tq.set_description('model {0}'.format(Path(model_path).stem))

        with torch.no_grad():
           for batch_num, (inputs, paths) in enumerate(loader):
                inputs = inputs.to(device)
                if tta:
                    y_pred = model.tta_flip(inputs).cpu().data.numpy() # use tta_flip
                else:
                    y_pred = torch.sigmoid(model(inputs)).data.cpu().numpy().astype(float)

                all_predictions.append(y_pred)
                tq.update(batch_size)

        tq.close()

        all_predictions = np.vstack(all_predictions)[:,0,:,:]

        top = 13
        left = 13
        bottom = top + 101
        right = left + 101

        all_predictions = all_predictions[:, top:bottom, left:right]
        fold_predictions.append(all_predictions)



    if len(fold_predictions) > 1:
        fold_predictions = np.stack(fold_predictions, axis=0)
        fold_predictions = np.mean(fold_predictions, axis=0)
    else:
        fold_predictions = fold_predictions[0]



    np.save(str(to_path / (Path('fold_{}.npy'.format(fold_no)))), fold_predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_path', type=str, default='data/models/', help='path to model folder')
    arg('--model_type', type=str, default='SE_ResNext50', help='network architecture',
        choices=['ResNet34','SE_ResNext50'])
    arg('--output_path', type=str, help='path to save images', default='data/predictions/')
    arg('--batch-size', type=int, default=64)
    arg('--workers', type=int, default=6)
    arg('--tta', type=int, default=1)
    arg('--fold', type=int, default = -1)
    arg('--config', default = 'SE_ResNext50_finetune.json')

    args = parser.parse_args()

    cfg = json.load(open('configs/' + args.config))

    # just convenient way to access config items by config.item in opposite to config['item']
    Config = namedtuple("Config", cfg.keys())
    config = Config(**cfg)

    test_path = os.path.join(data_path, 'test', 'images')
    ids = next(os.walk(test_path))[2]

    file_names = [i[:-4] for i in ids]

    print('num file_names = {}'.format(len(file_names)))

    output_path = Path(args.output_path) / args.model_type / 'test'
    output_path.mkdir(exist_ok=True, parents=True)
    print("Output path: {}".format(str(output_path)))

    if args.fold != -1:
        model, model_paths = get_model_and_paths(Path(args.model_path) / args.model_type, model_type=args.model_type,
                                                 fold_no=args.fold)
        predict(model, config, model_paths, file_names, args.batch_size, output_path, tta = args.tta, fold_no=args.fold)
    else:
        for f in range(5):
            print("Prediction fold: {}".format(f))
            model, model_paths = get_model_and_paths(Path(args.model_path) / args.model_type,
                                                     model_type=args.model_type, fold_no=f)
            predict(model, config, model_paths, file_names, args.batch_size, to_path=output_path, tta=args.tta, fold_no=f)