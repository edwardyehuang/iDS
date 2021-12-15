# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import os
import numpy as np
import tensorflow as tf

ANNOTATIONS_DIR = "Annotations"
INAGE_DIR = "JPEGImages"
LABEL_DIR = "SegmentationClass"

SEGMENTATION_LIST_DIR = "ImageSets/Segmentation"
SEGMENTATION_TRAIN_FILENAME = "train.txt"
SEGMENTATION_EVAL_FILENAME = "val.txt"
SEGMENTATION_TEST_FILENAME = "test.txt"

IMAGE_FILE_EXTENSION = ".jpg"
LABEL_FILE_EXTENSION = ".png"


from .dataset import Dataset


class VOCTest(Dataset):
    def __init__(self, dataset_dir):

        super().__init__(dataset_dir)

        self.ignore_label = 255
        self.num_class = 21

    def load_data_paths(self, dataset_dir):

        image_dir = os.path.join(dataset_dir, INAGE_DIR)
        test_list_path = os.path.join(dataset_dir, SEGMENTATION_LIST_DIR, SEGMENTATION_TEST_FILENAME)

        return self.__get_data_paths(test_list_path, image_dir), None

    def load_tf_data(self):

        (test_images_paths, _), _ = self.load_data()

        return (self.load_and_process_tensor_ds_from_path(test_images_paths, None, is_training=False), None)

    def __get_data_paths(self, names_list_path, images_dir):

        with open(names_list_path, "r") as f:
            images_names = f.read().split()

        images_paths = [os.path.join(images_dir, image_name + IMAGE_FILE_EXTENSION) for image_name in images_names]

        return images_paths, None

    # Copied from the deeplab offical source

    def create_label_colormap(self):

        _DATASET_MAX_ENTRIES = 256
        colormap = np.zeros((_DATASET_MAX_ENTRIES, 3), dtype=int)
        ind = np.arange(_DATASET_MAX_ENTRIES, dtype=int)

        for shift in reversed(range(8)):
            for channel in range(3):
                colormap[:, channel] |= self.__bit_get(ind, channel) << shift
            ind >>= 3

        return colormap

    def __bit_get(self, val, idx):

        return (val >> idx) & 1
