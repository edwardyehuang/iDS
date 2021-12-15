# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import os
import tensorflow as tf

TRAIN_IMAGE_DIR = "train2017"
TRAIN_LABEL_DIR = "train2017class"

EVAL_IMAGE_DIR = "val2017"
EVAL_LABEL_DIR = "val2017class"

SEGMENTATION_TRAIN_FILENAME = "train.txt"
SEGMENTATION_EVAL_FILENAME = "val.txt"

IMAGE_FILE_EXTENSION = ".jpg"
LABEL_FILE_EXTENSION = ".png"

from .dataset import Dataset


class COCO2017(Dataset):
    def __init__(self, dataset_dir):

        super(COCO2017, self).__init__(dataset_dir)

        self.ignore_label = 255
        self.num_class = 21

    def load_data_paths(self, dataset_dir):

        train_image_dir = os.path.join(dataset_dir, TRAIN_IMAGE_DIR)
        train_label_dir = os.path.join(dataset_dir, TRAIN_LABEL_DIR)

        eval_image_dir = os.path.join(dataset_dir, EVAL_IMAGE_DIR)
        eval_label_dir = os.path.join(dataset_dir, EVAL_LABEL_DIR)

        train_list_path = os.path.join(dataset_dir, SEGMENTATION_TRAIN_FILENAME)
        val_list_path = os.path.join(dataset_dir, SEGMENTATION_EVAL_FILENAME)

        return (
            self.__get_data_paths(train_list_path, train_image_dir, train_label_dir),
            self.__get_data_paths(val_list_path, eval_image_dir, eval_label_dir),
        )

    def __get_data_paths(self, names_list_path, images_dir, labels_dir):

        with open(names_list_path, "r") as f:
            images_names = f.read().split()

        images_paths = [os.path.join(images_dir, image_name + IMAGE_FILE_EXTENSION) for image_name in images_names]
        labels_paths = [os.path.join(labels_dir, image_name + LABEL_FILE_EXTENSION) for image_name in images_names]

        return images_paths, labels_paths
