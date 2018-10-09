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
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--predictions_path', type=str, default='data/predictions/SE_ResNext50/test',
        help='path where predicted images are located')

    args = parser.parse_args()



    test_path = os.path.join(data_path, 'test', 'images')
    img_count = len(list(Path(test_path).glob('*.png')))
    predictions = np.zeros((img_count,101,101))
    models_count = len(list(Path(args.predictions_path).glob('*.npy')))

    print("Models count:{}".format(models_count))

    print("Combining predictions from different models...")
    for model_name in tqdm(Path(args.predictions_path).glob('*.npy'), total = models_count):
        y_pred = np.load(str(model_name))
        #print(y_pred.shape)
        predictions += y_pred / models_count

    test_path = os.path.join(data_path, 'test', 'images')
    ids = next(os.walk(test_path))[2]
    file_names = [i[:-4] for i in ids]

    print("Creating submission file...")
    pred_dict = {}
    for i, file_name in tqdm(enumerate(file_names), total = img_count):
        y_pred = (predictions[i] > 0.45).astype(np.uint8)

        if y_pred.sum() < 10:
            y_pred[:, :] = 0

        y_pred = remove_small_holes(y_pred)
        y_pred = remove_small_objects(y_pred, min_size=16)

        pred_dict[file_name] = rle_encode(y_pred)



    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv('submission_9.csv')

    #os.system('kaggle competitions  submit -c  tgs-salt-identification-challenge -f  submission_9.csv -m sub1')


