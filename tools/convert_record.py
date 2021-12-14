# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import os, sys

rootpath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

sys.path.insert(1, rootpath)

import tensorflow as tf
import iseg.static_strings as ss

from absl import app, flags
from common_flags import FLAGS

from loaders.dataset_loader import dataset_name_to_dataset

flags.DEFINE_multi_string("convert_datasets", [ss.VOC2012], "name of the datasets")

flags.DEFINE_multi_string("tfrecord_outputs", None, "outputs of tfrecord")

flags.DEFINE_bool("compress", False, "Compress tfrecord")

flags.DEFINE_float("size_split", 8e9, "max size of each tfrecord file")


def main(argv):

    dataset_names = FLAGS.convert_datasets
    output_paths = FLAGS.tfrecord_outputs

    if len(dataset_names) != len(output_paths):
        raise ValueError("Num of datasets and output path does not matched")

    for i in range(len(dataset_names)):
        name = dataset_names[i]
        output_dir = output_paths[i]

        dataset = dataset_name_to_dataset(name)
        dataset.save_tf_record(output_dir, compress=FLAGS.compress, size_split=FLAGS.size_split)


if __name__ == "__main__":
    app.run(main)
