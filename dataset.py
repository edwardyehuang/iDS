# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import iseg.static_strings as ss
import tensorflow as tf
from PIL import Image

import ids.tfrecordutil as tfrecordutil

from iseg.data_process.pipeline import StandardAugmentationsPipeline
from iseg.utils.data_loader import load_image_tensor_from_path


class Dataset(object):
    @property
    def dataset_dir(self):
        return self.__dataset_dir

    def __init__(self, dataset_dir):
        self.__dataset_dir = dataset_dir

        self.mean_pixel = [127.5, 127.5, 127.5]

        self.ignore_label = 255
        self.num_class = 21
        self.val_image_count = 0
        self.class_weights=None

        self.crop_height = 513
        self.crop_width = 513

        self.eval_crop_height = None
        self.eval_crop_width = None

        self.prob_of_flip = 0.5
        self.min_scale_factor = 0.5
        self.max_scale_factor = 2.0
        self.scale_factor_step_size = 0.25

        self.min_resize_value = None
        self.max_resize_value = None

        self.random_brightness = True
        self.photo_metric_distortion = False

        self.swap_trainval = False

        self.compress = False
        self.trainval = False

        self.shuffle_raw_image_paths = False

        self.use_tfrecord = True

        self.__train_augments_pipeline = None
        self.__val_augments_pipeline = None

    def load_data_paths(self, dataset_dir):
        raise RuntimeError("You shoud not call the base class")

    def load_tf_data(self):

        train_ds = None
        val_ds = None

        if not self.use_tfrecord:
            train_ds, val_ds = self.load_trainval_tensor_ds()
        else:
            if not isinstance(self.__dataset_dir, str):
                raise ValueError('Path of TFRecord must be "str", not "{}"'.format(type(self.__dataset_dir)))

            train_ds = self.read_tf_record(True)
            val_ds = self.read_tf_record(False)

        if self.swap_trainval:
            tmp = train_ds
            train_ds = val_ds
            val_ds = tmp

        if self.trainval:
            train_ds = train_ds.concatenate(val_ds)

        train_ds = self.process_tensor_ds(train_ds, True)
        val_ds = self.process_tensor_ds(val_ds, False)

        return train_ds, val_ds

    def read_tf_record(self, training=False):
        if training:
            return tfrecordutil.read_tesnor_ds_from_tfrecords_dir(
                self._tfrecord_read_map_fn, self.__dataset_dir, ss.TRAIN, compress=self.compress
            )
        else:
            return tfrecordutil.read_tesnor_ds_from_tfrecords_dir(
                self._tfrecord_read_map_fn, self.__dataset_dir, ss.VAL, compress=self.compress
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

        return load_image_tensor_from_path(image_path, label_path)


    def process_tensor_ds(self, ds, is_training=False):

        if is_training:
            if not self.__train_augments_pipeline:
                self.__train_augments_pipeline = self.create_augment_pipeline(True)

            return self.__train_augments_pipeline(ds)
        else:
            if not self.__val_augments_pipeline:
                self.__val_augments_pipeline = self.create_augment_pipeline(False)

            return self.__val_augments_pipeline(ds)

    def create_augment_pipeline(self, training=False):

        return StandardAugmentationsPipeline(
            training=training,
            mean_pixel=self.mean_pixel,
            ignore_label=self.ignore_label,
            min_resize_value=self.min_resize_value,
            max_resize_value=self.max_resize_value,
            crop_height=self.crop_height,
            crop_width=self.crop_width,
            eval_crop_height=self.eval_crop_height,
            eval_crop_width=self.eval_crop_width,
            prob_of_flip=self.prob_of_flip,
            min_scale_factor=self.min_scale_factor,
            max_scale_factor=self.max_scale_factor,
            scale_factor_step_size=self.scale_factor_step_size,
            random_brightness=self.random_brightness,
            photo_metric_distortions=self.photo_metric_distortion,
        )
