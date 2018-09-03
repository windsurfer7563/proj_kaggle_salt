import numpy as np
import torch
from torch.utils.data import Dataset
import os
from skimage.io import imread
from torchvision.transforms import ToTensor, Normalize, Compose

data_path = 'data'

train_image_dir = os.path.join(data_path, 'train')
test_image_dir = os.path.join(data_path, 'test')


class SaltDataset(Dataset):
    def __init__(self, img_ids, transform=None, mode='train'):
        self.image_ids =  img_ids
        self.transform = transform
        self.mode = mode
        self.img_transform = Compose([
        ToTensor(),
        #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_file_name = self.image_ids[idx] + '.png'
        if self.mode == 'train':
            img_path = os.path.join(train_image_dir,'images', img_file_name)
            mask_path = os.path.join(train_image_dir,'masks',  img_file_name)
            mask = imread(mask_path)
        else:
            img_path = os.path.join(test_image_dir, 'images', img_file_name)

        img = imread(img_path)

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        if self.mode == 'train':
            return self.img_transform(img), torch.from_numpy(np.expand_dims(mask, axis=0).astype(float)).float()
        else:
            return self.img_transform(img), str(img_file_name)