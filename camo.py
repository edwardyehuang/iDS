# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import os
import tensorflow as tf
import numpy as np

from PIL import Image

INAGE_DIR = "Image"
LABEL_DIR = "GT_Object"

TRAIN_DIR = "Train"
TEST_DIR = "Test"

SEGMENTATION_TRAIN_FILENAME = "train.txt"
SEGMENTATION_EVAL_FILENAME = "test.txt"

IMAGE_FILE_EXTENSION = ".jpg"
LABEL_FILE_EXTENSION = ".png"

from .dataset import Dataset



class CAMO(Dataset):
    def __init__(self, dataset_dir):

        super().__init__(dataset_dir)

        self.ignore_label = 255
        self.num_class = 2
        self.val_image_count = 4000
        self.compress = True

        self.max_resize_height = 512
        self.max_resize_width = 512

    def load_data_paths(self, dataset_dir):

        train_dir = os.path.join(dataset_dir, TRAIN_DIR)
        test_dir = os.path.join(dataset_dir, TEST_DIR)

        train_image_dir = os.path.join(train_dir, INAGE_DIR)
        train_label_dir = os.path.join(train_dir, LABEL_DIR)

        test_image_dir = os.path.join(test_dir, INAGE_DIR)
        test_label_dir = os.path.join(test_dir, LABEL_DIR)

        return (
            self.__get_data_paths(train_image_dir, train_label_dir),
            self.__get_data_paths(test_image_dir, test_label_dir),
        )

    def __get_data_paths(self, image_dir, label_dir):

        image_names = os.listdir(image_dir)
        
        images_paths = []
        labels_paths = []

        for i in range(len(image_names)):
            image_name = image_names[i]
            image_path = os.path.join(image_dir, image_name)
            
            label_name = str.replace(image_name, IMAGE_FILE_EXTENSION, LABEL_FILE_EXTENSION)
            label_path = os.path.join(label_dir, label_name)

            if not os.path.exists(label_path):
                raise ValueError(f"Label file {label_path} does not exist.")
            
            images_paths.append(image_path)
            labels_paths.append(label_path)
            
        return images_paths, labels_paths


    def load_tensor_from_path(self, image_path, label_path):
        image_tensor, label_tensor = super().load_tensor_from_path(image_path, label_path)

        if label_tensor is not None:
            label_tensor = tf.cast(label_tensor > 0, tf.int32)

        return image_tensor, label_tensor
