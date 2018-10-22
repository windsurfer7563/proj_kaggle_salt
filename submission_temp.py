from pathlib import Path
import argparse
from models.dataset import data_path
from skimage.io import imread
from skimage import img_as_float
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

from skimage.morphology import remove_small_holes, remove_small_objects, binary_erosion, binary_dilation, disk, closing, opening

def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


if __name__ == '__main__':


    test_path = os.path.join(data_path, 'test', 'images')
    img_count = len(list(Path(test_path).glob('*.png')))
    predictions = np.zeros((img_count, 101, 101))
    predictions_img = np.zeros(img_count)


    print("Combining predictions from different models...")
    #y_pred_867 = np.load('model_867.npy')
    y_pred_858 = np.load('model_858.npy')
    y_pred_857 = np.load('model_857.npy')
    #predictions_img = np.load('model_867_img.npy')

    #y_pred_858[predictions_img < 0.4] = 0
    #y_pred_857[predictions_img < 0.4] = 0


    predictions = 0.5 * y_pred_858 + 0.5 * y_pred_857


    test_path = os.path.join(data_path, 'test', 'images')
    ids = next(os.walk(test_path))[2]
    file_names = [i[:-4] for i in ids]


    print("Creating submission file...")
    pred_dict = {}
    for i, file_name in tqdm(enumerate(file_names), total = img_count):
        y_pred = (predictions[i] > 0.4).astype(np.uint8)

        #if y_pred.sum() < 10:
        #    y_pred[:, :] = 0

        y_pred = remove_small_holes(y_pred)
        #y_pred = remove_small_objects(y_pred, min_size=16)

        pred_dict[file_name] = rle_encode(y_pred)


    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv('submission_optimistic.csv')

        #os.system('kaggle competitions  submit -c  tgs-salt-identification-challenge -f  submission_70_15_15.csv -m 70_15_15')


