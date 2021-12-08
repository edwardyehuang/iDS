# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import os, sys

rootpath = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, os.pardir))

sys.path.insert(1, rootpath)

import numpy as np
import iseg.static_strings as ss
import tensorflow as tf
import ids.tfrecordutil as tfrecordutil

from PIL import Image

from ids.cityscapes_fine import get_colormap

def get_ds_from_dir (input_dir, input_prefix = ss.TRAIN):
    
    return tfrecordutil.read_tesnor_ds_from_tfrecords_dir(input_dir, input_prefix)


if __name__ == "__main__":
    
    path = "/data2/edwardyehuang/Dataset/tfrecords/cityscapes_fine"

    ds = get_ds_from_dir(path)

    colormap = get_colormap().astype(np.uint8)

    for image, label in ds.skip(56).take(1):

        image = tf.cast(image, tf.int32)
        image = image.numpy().astype(np.uint8)
        image = Image.fromarray(image, mode = "RGB")

        image.show()

        label = tf.cast(label, tf.int32)
        label = tf.squeeze(label, axis = -1)
        label = label.numpy().astype(np.uint8)
        label = Image.fromarray(label, mode = "P")
        label.putpalette(colormap)

        # label.show()


        input()


