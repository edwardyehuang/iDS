# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import os
import tensorflow as tf
import numpy as np

ANNOTATIONS_DIR = "Annotations"
INAGE_DIR = "JPEGImages"
LABEL_DIR = "SegmentationClass"

SEGMENTATION_LIST_DIR = "ImageSets/Segmentation"
SEGMENTATION_TRAIN_FILENAME = "train.txt"
SEGMENTATION_EVAL_FILENAME = "val.txt"
SEGMENTATION_TRAINVAL_FILENAME = "hardtrainval.txt"

IMAGE_FILE_EXTENSION = ".jpg"
LABEL_FILE_EXTENSION = ".png"

from .dataset import Dataset


class Voc2012(Dataset):
    def __init__(self, dataset_dir):

        super().__init__(dataset_dir)

        self.ignore_label = 255
        self.num_class = 21
        self.val_image_count = 1449

    def load_data_paths(self, dataset_dir):

        image_dir = os.path.join(dataset_dir, INAGE_DIR)
        label_dir = os.path.join(dataset_dir, LABEL_DIR)

        train_list_path = os.path.join(dataset_dir, SEGMENTATION_LIST_DIR, SEGMENTATION_TRAIN_FILENAME)
        val_list_path = os.path.join(dataset_dir, SEGMENTATION_LIST_DIR, SEGMENTATION_EVAL_FILENAME)

        return (
            self.__get_data_paths(train_list_path, image_dir, label_dir),
            self.__get_data_paths(val_list_path, image_dir, label_dir),
        )

    def __get_data_paths(self, names_list_path, images_dir, labels_dir):

        with open(names_list_path, "r") as f:
            images_names = f.read().split()

        images_paths = [os.path.join(images_dir, image_name + IMAGE_FILE_EXTENSION) for image_name in images_names]
        labels_paths = [os.path.join(labels_dir, image_name + LABEL_FILE_EXTENSION) for image_name in images_names]

        return images_paths, labels_paths

    def create_label_colormap(self):
        return get_colormap()


# Copied from the deeplab offical source
def get_colormap():
    _DATASET_MAX_ENTRIES = 256
    colormap = np.zeros((_DATASET_MAX_ENTRIES, 3), dtype=int)
    ind = np.arange(_DATASET_MAX_ENTRIES, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap


def bit_get(val, idx):
    return (val >> idx) & 1
