from pathlib import Path
import argparse
from models.dataset import data_path
from skimage.io import imread
from skimage import img_as_float
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import pickle

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

    neib_df = pickle.load(open('neib_df2.pkl', 'rb'))

    test_path = os.path.join(data_path, 'test', 'images')
    img_count = len(list(Path(test_path).glob('*.png')))
    predictions = np.zeros((img_count, 101, 101))
    predictions_img = np.zeros(img_count)


    print("Combining predictions from different models...")
    y_pred_867 = np.load('model_867.npy')
    y_pred_858 = np.load('model_858.npy')
    y_pred_857 = np.load('model_857.npy')
    predictions_img = np.load('model_867_img.npy')

    #y_pred_858[predictions_img < 0.4] = 0
    #y_pred_857[predictions_img < 0.4] = 0


    predictions = 0.7 * y_pred_867 + 0.2 * y_pred_858 + 0.1 * y_pred_857


    test_path = os.path.join(data_path, 'test', 'images')
    ids = next(os.walk(test_path))[2]
    file_names = [i[:-4] for i in ids]


    print("Creating submission file...")
    pred_dict = {}
    corr_no = 0

    for i, file_name in tqdm(enumerate(file_names), total = img_count):

        id_ = file_name
        p = 0.4775
        if id_ in neib_df.index:
            corr = neib_df.loc[id_, 'max_corr']
            if corr > 0.03:
                p = 0
                corr_no += 1
            else:
                zero_corr_no = (neib_df.loc[id_, ['left_corr', 'right_corr', 'top_corr', 'bottom_corr']] == 0).sum()
                if zero_corr_no == 4:
                    predictions_img[i] = predictions_img[i] * 0.5


        predictions[i] = predictions[i] * (predictions_img[i] >= p)

        y_pred = (predictions[i] > 0.42).astype(np.uint8)

        #if y_pred.sum() < 10:
        #    y_pred[:, :] = 0

        y_pred = remove_small_holes(y_pred)
        #y_pred = remove_small_objects(y_pred, min_size=10)

        pred_dict[file_name] = rle_encode(y_pred)

    print("Found {} neigbors".format(corr_no))
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv('submission_70_20_10_new.csv')

        #os.system('kaggle competitions  submit -c  tgs-salt-identification-challenge -f  submission_70_15_15.csv -m 70_15_15')


