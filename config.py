from collections import namedtuple

Config = namedtuple("Config", [
    "dataset_path",
    "iter_size",
    "folder",
    "target_rows",
    "target_cols",
    "num_channels",
    "model",
    "loss",
    "optimizer",
    "lr",
    "lr_steps",
    "lr_gamma",
    "warmap_lr",
    'min_lr',
    'initial_lr',
    "batch_size",
    "epoch_size",
    "nb_epoch",
    "predict_batch_size",
    "test_pad",
    "results_dir",
    "num_classes",
    "ignore_target_size",
    "warmup",
    "scheduler"
])