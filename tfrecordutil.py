# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import sys
import os

import tensorflow as tf

import gc


def read_tesnor_ds_from_tfrecords_dir(
    example_mapping_fn, input_dir, input_prefix, input_ext=".tfrecord", compress=False
):
    matched_files = []

    for f in tf.io.gfile.listdir(input_dir):
        if input_prefix in f and input_ext in f:
            matched_files.append(os.path.join(input_dir, f))

    dataset = None

    compress = "ZLIB" if compress else None

    for path in matched_files:
        ds = tf.data.TFRecordDataset(path, compression_type=compress, num_parallel_reads=tf.data.experimental.AUTOTUNE)

        if dataset is None:
            dataset = ds
        else:
            dataset = dataset.concatenate(ds)

    if dataset is not None:
        dataset = dataset.map(example_mapping_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset


def save_tensor_ds_to_tfrecord(
    ds, example_mapping_fn, output_dir, output_prefix, output_ext=".tfrecord", size_split=4e9, compress=False
):

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    count = 0
    bytes_count = 0
    split_count = 0

    # ds = ds.map(map_ds_to_example_string)

    output_path_with_prefix = os.path.join(output_dir, output_prefix)

    processed_list = []
    bytes_count = 0

    def save_tfrecord_to_path(data, path):
        __save_converted_string(data, path, compress=compress)

    for tensors in ds:

        converted_string = map_ds_to_example_string(tensors, example_mapping_fn)
        processed_list.append(converted_string)

        bytes_count += sys.getsizeof(converted_string.numpy())
        del converted_string

        count += 1
        print("Converted {}".format(count))

        if bytes_count >= size_split:
            path = "{}-{}{}".format(output_path_with_prefix, split_count, output_ext)
            save_tfrecord_to_path(processed_list, path)

            split_count += 1
            processed_list.clear()
            del processed_list

            bytes_count = 0

            gc.collect()

            processed_list = []

    path = "{}-{}{}".format(output_path_with_prefix, split_count, output_ext)
    save_tfrecord_to_path(processed_list, path)


def __save_converted_string(processed_list, output_path, compress=False):
    ds = tf.data.Dataset.from_tensor_slices(processed_list)

    option = "ZLIB" if compress else None

    writer = tf.data.experimental.TFRecordWriter(output_path, compression_type=option)
    writer.write(ds)


def map_ds_to_example_string(tensors, example_mapping_fn):
    example = tf.train.Example(features=tf.train.Features(feature=example_mapping_fn(*tensors)))
    example = example.SerializeToString()

    return tf.reshape(example, ())


def bytes_feature(tensor):
    data = tf.io.serialize_tensor(tensor)
    data = data.numpy()

    feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))

    return feature


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
