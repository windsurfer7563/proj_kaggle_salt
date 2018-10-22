import numpy as np
import torch
from torch.utils.data import Dataset
import os
from skimage.io import imread
from torchvision.transforms import ToTensor, Normalize, Compose
from albumentations import Compose as AlbuCompose, PadIfNeeded, Resize
import cv2
from skimage.morphology import dilation, disk

data_path = 'data'

train_image_dir = os.path.join(data_path, 'train')
test_image_dir = os.path.join(data_path, 'test')


class SaltDataset(Dataset):
    def __init__(self, img_ids, config, transform=None, mode='train'):
        self.image_ids = img_ids
        self.transform = transform
        self.mode = mode
        paddings = {'replicate': cv2.BORDER_REPLICATE, 'reflect': cv2.BORDER_REFLECT_101, 'combine': 'combine', 'none': None}
        self.padding = paddings[config.padding]
        self.use_depth = config.use_depth
        self.img_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.5, 0.406], std=[0.229, 0.292, 0.225]),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img_file_name = self.image_ids[idx] + '.png'
        if self.mode == 'train':
            img_path = os.path.join(train_image_dir,'images', img_file_name)
            mask_path = os.path.join(train_image_dir,'masks',  img_file_name)
            mask = imread(mask_path)
            mask = (mask > 0).astype(float)
        elif self.mode == 'predict':
            img_path = os.path.join(train_image_dir, 'images', img_file_name)
            mask = None
        else: #predict for test images
            img_path = os.path.join(test_image_dir, 'images', img_file_name)
            mask = None

        img = imread(img_path)

        if self.transform is not None:
            if mask is not None:
                data = {"image": img, "mask": mask}
                augmented = self.transform(**data)
                img, mask = augmented["image"], augmented["mask"]
                mask = (mask > 0).astype(float)
            else:
                data = {"image": img}
                augmented = self.transform(**data)
                img = augmented["image"]

        if self.padding is not None:
            if self.padding == 'combine':
                img = cv2.copyMakeBorder(img, 13, 14, 0, 0, borderType=cv2.BORDER_REPLICATE)
                img = cv2.copyMakeBorder(img, 0, 0, 13, 14, borderType=cv2.BORDER_REFLECT_101)
                if mask is not None:
                    mask = cv2.copyMakeBorder(mask, 13, 14, 0, 0, borderType=cv2.BORDER_REPLICATE)
                    mask = cv2.copyMakeBorder(mask, 0, 0, 13, 14, borderType=cv2.BORDER_REFLECT_101)
            else:
                img = cv2.copyMakeBorder(img, 13, 14, 13, 14, borderType=self.padding)
                if mask is not None:
                    mask = cv2.copyMakeBorder(mask, 13, 14, 13, 14, borderType=self.padding)
        else:
            img = cv2.resize(img,(128,128), interpolation = cv2.INTER_LINEAR)
            if mask is not None:
                mask = cv2.resize(mask,(128,128), interpolation = cv2.INTER_NEAREST)

        if self.use_depth:
            img = self.add_depth_channels(img)



            #if 0 < mask.sum() < 200:
            #    mask = dilation(mask, selem=disk(3))



        if self.mode == 'train':
            return self.img_transform(img), torch.from_numpy(np.expand_dims(mask, axis=0)).float()
        else:
            return self.img_transform(img), str(img_file_name)

    def add_depth_channels(self, img):
        h, w, _ = img.shape
        for row, const in enumerate(np.linspace(0, 1, h)):
            img[row, :, 1] = const
        img[:, :, 2] = img[:, :, 0] * img[:, :, 1]
        return img



    def get_by_name(self, file_name):
        idx = self.image_ids.index(file_name)
        return self[idx]