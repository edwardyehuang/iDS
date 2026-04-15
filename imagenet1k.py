import os

import numpy as np
import tensorflow as tf

from scipy.io import loadmat

from ids.dataset import Dataset
from iseg.data_process.pipeline import AugmentationsPipeLine


TRAIN_DIR = "train"
VAL_DIR = "val"
TRAIN_FILE_LIST = "file.txt"

DEVKIT_DIR = "ILSVRC2012_devkit_t12"
DEVKIT_META_PATH = os.path.join(DEVKIT_DIR, "data", "meta.mat")
DEVKIT_VAL_GT_PATH = os.path.join(DEVKIT_DIR, "data", "ILSVRC2012_validation_ground_truth.txt")


class ImageNet1KAugmentationsPipeLine(AugmentationsPipeLine):
	def __init__(self, process_fn, name=None):
		super().__init__(perform_post_process=False, name=name)
		self._process_fn = process_fn

	@tf.autograph.experimental.do_not_convert
	def process(self, image, label):
		return self._process_fn(image, label)


class ImageNet1K(Dataset):
	def __init__(self, dataset_dir):
		super().__init__(dataset_dir)

		self.num_class = 1000
		self.ignore_label = 255
		self.val_image_count = 50000

		# ImageNet classification models typically use 224x224 inputs.
		self.crop_height = 224
		self.crop_width = 224
		self.eval_crop_height = 224
		self.eval_crop_width = 224

		self._wnid_to_label = None
		self._train_augmentations_pipeline = None
		self._eval_augmentations_pipeline = None

	def load_data_paths(self, dataset_dir):
		train_dir = os.path.join(dataset_dir, TRAIN_DIR)
		val_dir = os.path.join(dataset_dir, VAL_DIR)

		wnid_to_label = self._load_wnid_to_label(dataset_dir)

		train_images, train_labels = self._load_train_data_paths(train_dir, wnid_to_label)
		val_images, val_labels = self._load_val_data_paths(dataset_dir, val_dir)

		return (train_images, train_labels), (val_images, val_labels)

	def _load_train_data_paths(self, train_dir, wnid_to_label):
		train_file_list = os.path.join(train_dir, TRAIN_FILE_LIST)

		image_paths = []
		labels = []

		if os.path.exists(train_file_list):
			with open(train_file_list, "r") as f:
				rel_paths = [x.strip() for x in f.readlines() if x.strip()]

			for rel_path in rel_paths:
				image_path = os.path.join(train_dir, rel_path)

				if not os.path.exists(image_path):
					raise ValueError(f"Image path {image_path} does not exist")

				wnid = rel_path.split("/", 1)[0]

				if wnid not in wnid_to_label:
					raise ValueError(f"WNID {wnid} not found in devkit meta mapping")

				image_paths.append(image_path)
				labels.append(wnid_to_label[wnid])
		else:
			wnids = sorted([
				x for x in os.listdir(train_dir)
				if os.path.isdir(os.path.join(train_dir, x))
			])

			for wnid in wnids:
				if wnid not in wnid_to_label:
					raise ValueError(f"WNID {wnid} not found in devkit meta mapping")

				wnid_dir = os.path.join(train_dir, wnid)
				filenames = sorted([
					x for x in os.listdir(wnid_dir)
					if x.lower().endswith(".jpeg")
				])

				for filename in filenames:
					image_paths.append(os.path.join(wnid_dir, filename))
					labels.append(wnid_to_label[wnid])

		if self.shuffle_raw_image_paths:
			shuffled_indices = np.random.permutation(len(image_paths))
			image_paths = [image_paths[i] for i in shuffled_indices]
			labels = [labels[i] for i in shuffled_indices]

		return image_paths, labels

	def _load_val_data_paths(self, dataset_dir, val_dir):
		gt_path = os.path.join(dataset_dir, DEVKIT_VAL_GT_PATH)

		if not os.path.exists(gt_path):
			raise ValueError(f"Validation ground-truth file does not exist: {gt_path}")

		image_names = sorted([
			x for x in os.listdir(val_dir)
			if x.lower().endswith(".jpeg")
		])
		image_paths = [os.path.join(val_dir, x) for x in image_names]

		with open(gt_path, "r") as f:
			# Ground-truth is 1-based ILSVRC2012_ID, convert to 0-based.
			labels = [int(x.strip()) - 1 for x in f.readlines() if x.strip()]

		if len(image_paths) != len(labels):
			raise ValueError(
				f"Validation image count {len(image_paths)} does not match label count {len(labels)}"
			)

		invalid_labels = [x for x in labels if x < 0 or x >= self.num_class]

		if len(invalid_labels) > 0:
			raise ValueError("Validation labels contain values outside [0, 999]")

		return image_paths, labels

	def _load_wnid_to_label(self, dataset_dir):
		if self._wnid_to_label is not None:
			return self._wnid_to_label

		meta_path = os.path.join(dataset_dir, DEVKIT_META_PATH)

		if not os.path.exists(meta_path):
			raise ValueError(f"Devkit meta.mat does not exist: {meta_path}")

		meta = loadmat(meta_path, squeeze_me=True, struct_as_record=False)

		if "synsets" not in meta:
			raise ValueError(f"Cannot find synsets in meta file: {meta_path}")

		synsets = np.atleast_1d(meta["synsets"])

		wnid_to_label = {}

		for synset in synsets:
			ilsvrc_id = int(self._get_mat_field(synset, "ILSVRC2012_ID"))

			if ilsvrc_id < 1 or ilsvrc_id > self.num_class:
				continue

			wnid = str(self._get_mat_field(synset, "WNID"))
			wnid_to_label[wnid] = ilsvrc_id - 1

		if len(wnid_to_label) != self.num_class:
			raise ValueError(
				f"Expected {self.num_class} leaf synsets in meta.mat, got {len(wnid_to_label)}"
			)

		self._wnid_to_label = wnid_to_label

		return self._wnid_to_label

	def _get_mat_field(self, mat_entry, field_name):
		if hasattr(mat_entry, field_name):
			return getattr(mat_entry, field_name)

		if isinstance(mat_entry, np.void) and mat_entry.dtype.names and field_name in mat_entry.dtype.names:
			return mat_entry[field_name]

		raise ValueError(f"Cannot read field {field_name} from MAT entry")

	def load_tensor_from_path(self, image_path, label):
		image_data = tf.io.read_file(image_path)

		image_tensor = tf.image.decode_jpeg(
			image_data,
			channels=3,
			dct_method="INTEGER_ACCURATE",
			try_recover_truncated=True,
		)

		image_tensor = tf.cast(image_tensor, tf.float32)
		label_tensor = tf.cast(label, tf.int32)

		return image_tensor, label_tensor

	def process_tensor_ds(self, ds, is_training=False):
		if ds is None:
			return ds

		if is_training:
			if self._train_augmentations_pipeline is None:
				self._train_augmentations_pipeline = self.create_augment_pipeline(training=True)

			return self._train_augmentations_pipeline(ds)

		if self._eval_augmentations_pipeline is None:
			self._eval_augmentations_pipeline = self.create_augment_pipeline(training=False)

		return self._eval_augmentations_pipeline(ds)

	def create_augment_pipeline(self, training=False):
		process_fn = self._train_process if training else self._eval_process
		pipeline_name = "imagenet1k_train_pipeline" if training else "imagenet1k_eval_pipeline"

		return ImageNet1KAugmentationsPipeLine(
			process_fn=process_fn,
			name=pipeline_name,
		)

	@tf.autograph.experimental.do_not_convert
	def _train_process(self, image, label):
		image = self._random_scale_for_train(image)
		image = self._pad_to_target_size(image, self.crop_height, self.crop_width)
		image = tf.image.random_crop(image, [self.crop_height, self.crop_width, 3])

		if self.random_brightness:
			image = tf.image.random_brightness(image, max_delta=32.0)

		if self.prob_of_flip > 0:
			image = tf.cond(
				tf.random.uniform([]) <= self.prob_of_flip,
				lambda: tf.image.flip_left_right(image),
				lambda: image,
			)

		image.set_shape([self.crop_height, self.crop_width, 3])

		return image, label

	@tf.autograph.experimental.do_not_convert
	def _eval_process(self, image, label):
		target_h = self.eval_crop_height if self.eval_crop_height is not None else self.crop_height
		target_w = self.eval_crop_width if self.eval_crop_width is not None else self.crop_width

		image = self._resize_for_eval(image, target_h, target_w)
		image.set_shape([target_h, target_w, 3])

		return image, label

	def _random_scale_for_train(self, image):
		image_shape = tf.shape(image)

		scale = self._sample_scale()

		scaled_h = tf.cast(tf.cast(image_shape[0], tf.float32) * scale, tf.int32)
		scaled_w = tf.cast(tf.cast(image_shape[1], tf.float32) * scale, tf.int32)

		scaled_h = tf.maximum(scaled_h, self.crop_height)
		scaled_w = tf.maximum(scaled_w, self.crop_width)

		image = tf.image.resize(image, [scaled_h, scaled_w], method="bilinear")

		return image

	def _sample_scale(self):
		raw_min_scale = tf.constant(self.min_scale_factor, dtype=tf.float32)
		raw_max_scale = tf.constant(self.max_scale_factor, dtype=tf.float32)

		min_scale = tf.minimum(raw_min_scale, raw_max_scale)
		max_scale = tf.maximum(raw_min_scale, raw_max_scale)

		if self.scale_factor_step_size is None or self.scale_factor_step_size <= 0:
			return tf.random.uniform([], minval=min_scale, maxval=max_scale)

		step = tf.constant(self.scale_factor_step_size, dtype=tf.float32)
		num_steps = tf.cast(tf.math.floor((max_scale - min_scale) / step), tf.int32) + 1
		rand_step = tf.random.uniform([], minval=0, maxval=num_steps, dtype=tf.int32)

		return tf.clip_by_value(min_scale + tf.cast(rand_step, tf.float32) * step, min_scale, max_scale)

	def _pad_to_target_size(self, image, target_h, target_w):
		image_shape = tf.shape(image)
		current_h = image_shape[0]
		current_w = image_shape[1]

		out_h = current_h + tf.maximum(target_h - current_h, 0)
		out_w = current_w + tf.maximum(target_w - current_w, 0)

		mean_pixel = tf.reshape(tf.cast(self.mean_pixel, image.dtype), [1, 1, 3])
		image = image - mean_pixel
		image = tf.image.pad_to_bounding_box(image, 0, 0, out_h, out_w)
		image = image + mean_pixel

		return image

	def _resize_for_eval(self, image, target_h, target_w):
		image_shape = tf.shape(image)

		scale_h = tf.cast(target_h, tf.float32) / tf.cast(image_shape[0], tf.float32)
		scale_w = tf.cast(target_w, tf.float32) / tf.cast(image_shape[1], tf.float32)
		scale = tf.maximum(scale_h, scale_w)

		resized_h = tf.cast(tf.round(tf.cast(image_shape[0], tf.float32) * scale), tf.int32)
		resized_w = tf.cast(tf.round(tf.cast(image_shape[1], tf.float32) * scale), tf.int32)

		image = tf.image.resize(image, [resized_h, resized_w], method="bilinear")
		image = tf.image.resize_with_crop_or_pad(image, target_h, target_w)

		return image

	@tf.autograph.experimental.do_not_convert
	def _tfrecord_read_map_fn(self, example_proto):
		features = {
			"image": tf.io.FixedLenFeature([], tf.string, default_value=""),
			"label": tf.io.FixedLenFeature([], tf.string, default_value=""),
			"height": tf.io.FixedLenFeature([], tf.int64),
			"width": tf.io.FixedLenFeature([], tf.int64),
			"depth": tf.io.FixedLenFeature([], tf.int64),
		}

		features = tf.io.parse_single_example(example_proto, features)

		image = self.tfrecord_decode_image(features["image"])
		label = tf.io.parse_tensor(features["label"], tf.int32)

		image = tf.reshape(image, [features["height"], features["width"], features["depth"]])
		label = tf.reshape(label, [])

		return image, label
