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

IMAGE_FILE_EXTENSION = ".png"
LABEL_FILE_EXTENSION = ".png"

from .dataset import Dataset


class CityScapesFine(Dataset):
    def __init__(self, dataset_dir):

        super(CityScapesFine, self).__init__(dataset_dir)

        self.__label_color_map = get_colormap()

        self.ignore_label = 255
        self.num_class = 19
        self.val_image_count = 500

        self.class_weights = [
            0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
            1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
            1.0865, 1.0955, 1.0865, 1.1529, 1.0507]

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

        return get_colormap()


# Copied from the deeplab offical source
def get_colormap():

    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]

    return colormap
