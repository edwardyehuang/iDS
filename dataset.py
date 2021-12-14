# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import iseg.static_strings as ss
import tensorflow as tf
from PIL import Image

import ids.tfrecordutil as tfrecordutil

from iseg.data_process.pipeline import StandardArgumentsPipeline


class Dataset(object):

    MEAN_PXIEL = [127.5, 127.5, 127.5]

    NUM_CLASSES = 21

    IGNORE_LABEL = 255

    VAL_IMAGE_COUNT = 0

    CROP_HEIGHT = 513

    CROP_WIDTH = 513

    EVAL_CROP_HEIGHT = None

    EVAL_CROP_WIDTH = None

    IGNORE_LABEL = 255

    DEPRECATED_LABEL = []

    MIN_RESIZE_VALUE = None

    MAX_RESIZE_VALUE = None

    RESIZE_FACTOR = None

    PROB_OF_FLIP = 0.5

    MIN_SCALE_FACTOR = 0.5

    MAX_SCALE_FACTOR = 2.0

    SCALE_FACTOR_STEP_SIZE = 0.25

    RANDOM_BRIGHTNESS = True

    PHOTO_METRIC_DISTORTION = (False,)

    USE_TFRECORD = True

    TRAINVAL = False

    COMPRESS = False

    SWAP_TRAINVAL = False

    REVERSED_LABEL = False

    @property
    def dataset_dir(self):
        return self.__dataset_dir

    def __init__(self, dataset_dir):
        self.__dataset_dir = dataset_dir

        self.__train_arugments_pipeline = None
        self.__val_arguments_pipeline = None

    def load_data_paths(self, dataset_dir):
        raise RuntimeError("You shoud not call the base class")

    def load_tf_data(self):

        train_ds = None
        val_ds = None

        if not self.USE_TFRECORD:
            train_ds, val_ds = self.load_trainval_tensor_ds()
        else:
            if not isinstance(self.__dataset_dir, str):
                raise ValueError('Path of TFRecord must be "str", not "{}"'.format(type(self.__dataset_dir)))

            train_ds = self.read_tf_record(True)
            val_ds = self.read_tf_record(False)

        if self.SWAP_TRAINVAL:
            tmp = train_ds
            train_ds = val_ds
            val_ds = tmp

        if self.TRAINVAL:
            train_ds = train_ds.concatenate(val_ds)

        train_ds = self.process_tensor_ds(train_ds, True)
        val_ds = self.process_tensor_ds(val_ds, False)

        return train_ds, val_ds

    def read_tf_record(self, training=False):
        if training:
            return tfrecordutil.read_tesnor_ds_from_tfrecords_dir(
                self._tfrecord_read_map_fn, self.__dataset_dir, ss.TRAIN, compress=self.COMPRESS
            )
        else:
            return tfrecordutil.read_tesnor_ds_from_tfrecords_dir(
                self._tfrecord_read_map_fn, self.__dataset_dir, ss.VAL, compress=self.COMPRESS
            )

    def save_tf_record(self, output_dir, compress=False, size_split=8e9):

        train_ds, val_ds = self.load_trainval_tensor_ds()

        if train_ds is not None:
            tfrecordutil.save_tensor_ds_to_tfrecord(
                example_mapping_fn=self._tfrecord_write_map_fn,
                ds=train_ds,
                output_dir=output_dir,
                output_prefix=ss.TRAIN,
                compress=compress,
                size_split=size_split,
            )

        if val_ds is not None:
            tfrecordutil.save_tensor_ds_to_tfrecord(
                example_mapping_fn=self._tfrecord_write_map_fn,
                ds=val_ds,
                output_dir=output_dir,
                output_prefix=ss.VAL,
                compress=compress,
                size_split=size_split,
            )

    def _tfrecord_read_map_fn(self, example_proto):
        features = {
            ss.IMAGE: tf.io.FixedLenFeature([], tf.string, default_value=""),
            ss.LABEL: tf.io.FixedLenFeature([], tf.string, default_value=""),
            ss.HEIGHT: tf.io.FixedLenFeature([], tf.int64),
            ss.WIDTH: tf.io.FixedLenFeature([], tf.int64),
            ss.DEPTH: tf.io.FixedLenFeature([], tf.int64),
        }

        features = tf.io.parse_single_example(example_proto, features)
        image = tf.io.parse_tensor(features[ss.IMAGE], tf.float32)
        label = tf.io.parse_tensor(features[ss.LABEL], tf.int32)

        image = tf.reshape(image, [features[ss.HEIGHT], features[ss.WIDTH], features[ss.DEPTH]])
        label = tf.reshape(label, [features[ss.HEIGHT], features[ss.WIDTH], 1])

        return image, label

    def _tfrecord_write_map_fn(self, image_tensor, label_tensor):
        image_shape = tf.shape(image_tensor)

        features = dict()

        features[ss.IMAGE] = tfrecordutil.bytes_feature(image_tensor)
        features[ss.HEIGHT] = tfrecordutil.int64_feature(image_shape[0])
        features[ss.WIDTH] = tfrecordutil.int64_feature(image_shape[1])
        features[ss.DEPTH] = tfrecordutil.int64_feature(image_shape[2])

        if label_tensor is not None:
            features[ss.LABEL] = tfrecordutil.bytes_feature(label_tensor)

        return features

    def load_trainval_tensor_ds(self):
        training_paths, val_paths = self.load_data_paths(self.__dataset_dir)

        train_ds = val_ds = None

        if training_paths is not None:
            train_ds = self.load_tensor_ds_from_path(training_paths)

        if val_paths is not None:
            val_ds = self.load_tensor_ds_from_path(val_paths)

        return train_ds, val_ds

    def load_tensor_ds_from_path(self, paths):
        path_dataset = tf.data.Dataset.from_tensor_slices(tuple(paths))
        return path_dataset.map(self.load_tensor_from_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def load_tensor_from_path(self, image_path, label_path):

        image_tensor = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
        image_tensor = tf.cast(image_tensor, tf.float32)

        label_tensor = None

        if label_path is not None:
            label_tensor = self._load_label_to_tensor(label_path)

        return image_tensor, label_tensor

    def process_tensor_ds(self, ds, is_training=False):

        if is_training:
            if not self.__train_arugments_pipeline:
                self.__train_arugments_pipeline = self.create_argument_pipeline(True)

            return self.__train_arugments_pipeline(ds)
        else:
            if not self.__val_arguments_pipeline:
                self.__val_arguments_pipeline = self.create_argument_pipeline(False)

            return self.__val_arguments_pipeline(ds)

    def create_argument_pipeline(self, training=False):

        return StandardArgumentsPipeline(
            training=training,
            mean_pixel=self.MEAN_PXIEL,
            ignore_label=self.IGNORE_LABEL,
            min_resize_value=self.MIN_RESIZE_VALUE,
            max_resize_value=self.MAX_RESIZE_VALUE,
            crop_height=self.CROP_HEIGHT,
            crop_width=self.CROP_WIDTH,
            eval_crop_height=self.EVAL_CROP_HEIGHT,
            eval_crop_width=self.EVAL_CROP_WIDTH,
            prob_of_flip=self.PROB_OF_FLIP,
            min_scale_factor=self.MIN_SCALE_FACTOR,
            max_scale_factor=self.MAX_SCALE_FACTOR,
            scale_factor_step_size=self.SCALE_FACTOR_STEP_SIZE,
            random_brightness=self.RANDOM_BRIGHTNESS,
            photo_metric_distortions=self.PHOTO_METRIC_DISTORTION,
        )

    def _load_label_to_tensor(self, label_path):
        label_tensor = tf.py_function(self.__load_label_to_tensor_internel, [label_path], tf.int32)
        label_tensor.set_shape([None, None, 1])

        return label_tensor

    def __load_label_to_tensor_internel(self, path_tensor):

        label_path = path_tensor.numpy()
        label_image = Image.open(label_path)
        label_array = tf.keras.preprocessing.image.img_to_array(label_image, "channels_last")

        label_tensor = tf.convert_to_tensor(label_array)
        label_tensor = tf.cast(label_tensor, tf.int32)

        return label_tensor
