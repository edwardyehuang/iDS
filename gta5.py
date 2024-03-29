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

IMAGE_FILE_EXTENSION = ".png"
LABEL_FILE_EXTENSION = ".png"

from .dataset import Dataset


class GTA5(Dataset):
    def __init__(self, dataset_dir):

        super().__init__(dataset_dir)

        self.ignore_label = 255
        self.num_class = 19
        self.compress = True


    def load_data_paths(self, dataset_dir):

        image_dir = os.path.join(dataset_dir, INAGE_DIR)
        label_dir = os.path.join(dataset_dir, LABEL_DIR)

        train_list_path = os.path.join(dataset_dir, SEGMENTATION_TRAIN_FILENAME)

        return self.__get_data_paths(train_list_path, image_dir, label_dir), None

    def __get_data_paths(self, names_list_path, images_dir, labels_dir):

        with open(names_list_path, "r") as f:
            images_names = f.read().split()

        images_paths = [os.path.join(images_dir, image_name + IMAGE_FILE_EXTENSION) for image_name in images_names]
        labels_paths = [os.path.join(labels_dir, image_name + LABEL_FILE_EXTENSION) for image_name in images_names]

        return images_paths, labels_paths
