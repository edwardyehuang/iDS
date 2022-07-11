# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import os
import tensorflow as tf
import numpy as np

from PIL import Image

INAGE_DIR = "images"
LABEL_DIR = "labels"

SEGMENTATION_TRAIN_FILENAME = "train.txt"
SEGMENTATION_EVAL_FILENAME = "val.txt"

IMAGE_FILE_EXTENSION = ".jpg"
LABEL_FILE_EXTENSION = ".png"

from .dataset import Dataset


class PascalContext(Dataset):
    def __init__(self, dataset_dir, ignore_label_to_background=False):

        super().__init__(dataset_dir)

        self.ignore_label = 0
        self.num_class = 59
        self.val_image_count = 5105
        self.compress = True

        if ignore_label_to_background:
            self.num_class = 60
            self.ignore_label = 255

        self.__label_color_map = self.create_label_colormap()

    def load_data_paths(self, dataset_dir):

        image_dir = os.path.join(dataset_dir, INAGE_DIR)
        label_dir = os.path.join(dataset_dir, LABEL_DIR)

        train_list_path = os.path.join(dataset_dir, SEGMENTATION_TRAIN_FILENAME)
        val_list_path = os.path.join(dataset_dir, SEGMENTATION_EVAL_FILENAME)

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

        colormap = [
            (0, 0, 0),
            (128, 0, 0),
            (0, 128, 0),
            (128, 128, 0),
            (0, 0, 128),
            (128, 0, 128),
            (0, 128, 128),
            (128, 128, 128),
            (64, 0, 0),
            (192, 0, 0),
            (64, 128, 0),
            (192, 128, 0),
            (64, 0, 128),
            (192, 0, 128),
            (64, 128, 128),
            (192, 128, 128),
            (0, 64, 0),
            (128, 64, 0),
            (0, 192, 0),
            (128, 192, 0),
            (0, 64, 128),
            (128, 64, 128),
            (0, 192, 128),
            (128, 192, 128),
            (64, 64, 0),
            (192, 64, 0),
            (64, 192, 0),
            (192, 192, 0),
            (64, 64, 128),
            (192, 64, 128),
            (64, 192, 128),
            (192, 192, 128),
            (0, 0, 64),
            (128, 0, 64),
            (0, 128, 64),
            (128, 128, 64),
            (0, 0, 192),
            (128, 0, 192),
            (0, 128, 192),
            (128, 128, 192),
            (64, 0, 64),
            (192, 0, 64),
            (64, 128, 64),
            (192, 128, 64),
            (64, 0, 192),
            (192, 0, 192),
            (64, 128, 192),
            (192, 128, 192),
            (0, 64, 64),
            (128, 64, 64),
            (0, 192, 64),
            (128, 192, 64),
            (0, 64, 192),
            (128, 64, 192),
            (0, 192, 192),
            (128, 192, 192),
            (64, 64, 64),
            (192, 64, 64),
            (64, 192, 64),
            (192, 192, 64),
        ]

        colormap = np.array(colormap)

        return colormap
