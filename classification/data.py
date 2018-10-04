import numpy as np
import collections
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torchvision import transforms
from albumentations import (HorizontalFlip, VerticalFlip, Normalize,
    ShiftScaleRotate, Blur, OpticalDistortion,  GridDistortion, HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur,
    MedianBlur, IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast, RandomBrightness,
    Flip, OneOf, Compose, RandomGamma, PadIfNeeded, CLAHE, InvertImg, ElasticTransform, IAAPerspective, RandomSizedCrop,
    RandomCrop, Resize, DualTransform)


from prometheus.utils.parse import parse_in_csvs
from prometheus.utils.factory import UtilsFactory
from prometheus.data.reader import ImageReader, ScalarReader, ReaderCompose
from prometheus.data.augmentor import Augmentor
from prometheus.data.sampler import BalanceClassSampler
from prometheus.dl.datasource import AbstractDataSource

# ---- Augmentations ----

IMG_SIZE = 196


def strong_aug(p=1):
    return Compose([
        HorizontalFlip(p=0.5),
        OneOf([
            RandomCrop(94, 94,  p=0.6),
            ShiftScaleRotate(shift_limit=(0.1, 0.1), scale_limit=(0.05, 0.05), rotate_limit=10, p=0.4),
        ], p=0.6),
        OneOf([
            ElasticTransform(p=0.2, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            IAAPiecewiseAffine(p=.4),
            GridDistortion(p=0.4),
        ], p=.4),
        OneOf([
            # CLAHE(clip_limit=2),
            RandomGamma((90, 110)),
            IAAEmboss((0.1, 0.4), (0.1, 0.6)),
            RandomContrast(0.1),
            RandomBrightness(0.1),
        ], p=0.5),

    ], p=p)



AUG_TRAIN = strong_aug(p=0.85)
AUG_INFER = Compose([
    Resize(IMG_SIZE, IMG_SIZE),
    PadIfNeeded(224, 224),
    Normalize(),
])

TRAIN_TRANSFORM_FN = [
    Augmentor(
        dict_key="image",
        augment_fn=lambda x: AUG_TRAIN(image=x)["image"]),
]

INFER_TRANSFORM_FN = [
    Augmentor(
        dict_key="image",
        augment_fn=lambda x: AUG_INFER(image=x)["image"]),
    Augmentor(
        dict_key="image",
        augment_fn=lambda x: torch.tensor(x).permute(2, 0, 1)),
]


# ---- Data ----

class DataSource(AbstractDataSource):

    @staticmethod
    def prepare_transforms(*, mode, stage=None):
        if mode == "train":
            if stage in ["debug", "stage1"]:
                return transforms.Compose(
                    TRAIN_TRANSFORM_FN + INFER_TRANSFORM_FN)
            elif stage == "stage2":
                return transforms.Compose(INFER_TRANSFORM_FN)
        elif mode == "valid":
            return transforms.Compose(INFER_TRANSFORM_FN)
        elif mode == "infer":
            return transforms.Compose(INFER_TRANSFORM_FN)

    @staticmethod
    def prepare_loaders(args, data_params, stage=None):
        loaders = collections.OrderedDict()

        df, df_train, df_valid, df_infer = parse_in_csvs(data_params)

        open_fn = [
            ImageReader(
                row_key="filename", dict_key="image",
                datapath=data_params.get("datapath", None), grayscale=False),
            ScalarReader(
                row_key="class", dict_key="target",
                default_value=0, dtype=np.int64)
        ]
        open_fn = ReaderCompose(readers=open_fn)

        if len(df_train) > 0:
            labels = [x["class"] for x in df_train]
            #sampler = BalanceClassSampler(labels, mode="upsampling")
            sampler = None
            train_loader = UtilsFactory.create_loader(
                data_source=df_train,
                open_fn=open_fn,
                dict_transform=DataSource.prepare_transforms(
                    mode="train", stage=stage),
                dataset_cache_prob=getattr(args, "dataset_cache_prob", -1),
                batch_size=args.batch_size,
                workers=args.workers,
                shuffle=sampler is None,
                sampler=sampler)

            print("Train samples", len(train_loader) * args.batch_size)
            print("Train batches", len(train_loader))
            loaders["train"] = train_loader

        if len(df_valid) > 0:
            sampler = None

            valid_loader = UtilsFactory.create_loader(
                data_source=df_valid,
                open_fn=open_fn,
                dict_transform=DataSource.prepare_transforms(
                    mode="valid", stage=stage),
                dataset_cache_prob=-1,
                batch_size=args.batch_size,
                workers=args.workers,
                shuffle=False,
                sampler=sampler)

            print("Valid samples", len(valid_loader) * args.batch_size)
            print("Valid batches", len(valid_loader))
            loaders["valid"] = valid_loader

        if len(df_infer) > 0:
            infer_loader = UtilsFactory.create_loader(
                data_source=df_infer,
                open_fn=open_fn,
                dict_transform=DataSource.prepare_transforms(
                    mode="infer", stage=None),
                dataset_cache_prob=-1,
                batch_size=args.batch_size,
                workers=args.workers,
                shuffle=False,
                sampler=None)

            print("Infer samples", len(infer_loader) * args.batch_size)
            print("Infer batches", len(infer_loader))
            loaders["infer"] = infer_loader

        return loaders

