from pathlib import Path
import argparse
from skimage.io import imread
from skimage import img_as_float
import numpy as np
from tqdm import tqdm
import pandas as pd


def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--predictions_path', type=str, default='data/predictions/AlbuNet/test',
        help='path where predicted images are located')

    args = parser.parse_args()

    pred_dict = {}
    for file_name in tqdm((Path(args.predictions_path).glob('*'))):
        y_pred = (img_as_float(imread(str(file_name))) > 0.46).astype(np.uint8)

        if y_pred.sum() < 35:
            y_pred[:,:] = 0
        pred_dict[str(file_name.name)[0:-4]] = rle_encode(y_pred)

    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv('submission_1.csv')


