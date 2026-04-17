import os

import numpy as np
import tensorflow as tf

from scipy.io import loadmat

from ids.dataset import Dataset
from iseg.data_process.augments import (
	CenterCropImageAugment,
	RandAugmentImageAugment,
	RandomErasingAugment,
	RandomGaussianBlurImageAugment,
	RandomGrayscaleImageAugment,
	RandomBrightnessAugment,
	RandomFlipImageAugment,
	RandomJEPGQualityAugment,
	RandomNoisyEvalAugment,
	RandomPhotoMetricDistortions,
	RandomResizedCropImageAugment,
	ResizeShortEdgeImageAugment,
)
from iseg.data_process.pipeline import ClassificationAugmentationsPipeLine


TRAIN_DIR = "train"
VAL_DIR = "val"
TRAIN_FILE_LIST = "file.txt"

DEVKIT_DIR = "ILSVRC2012_devkit_t12"
DEVKIT_META_PATH = os.path.join(DEVKIT_DIR, "data", "meta.mat")
DEVKIT_VAL_GT_PATH = os.path.join(DEVKIT_DIR, "data", "ILSVRC2012_validation_ground_truth.txt")


class ImageNet1KAugmentationsPipeLine(ClassificationAugmentationsPipeLine):
	def __init__(self, target_height, target_width, augments=None, name=None):
		super().__init__(
			target_height=target_height,
			target_width=target_width,
			augments=augments if augments is not None else [],
			name=name,
		)


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
		self.classification_crop_area_range = (0.2, 1.0)
		self.classification_crop_aspect_ratio_range = (3.0 / 4.0, 4.0 / 3.0)
		self.classification_crop_max_attempts = 10
		self.eval_crop_pct = 0.875

		self.classification_use_rand_augment = False
		self.classification_rand_augment_num_layers = 2
		self.classification_rand_augment_magnitude = 9.0
		self.classification_rand_augment_magnitude_max = 10.0
		self.classification_rand_augment_magnitude_std = 0.0
		self.classification_rand_augment_op_prob = 1.0
		self.classification_rand_augment_max_rotate_degree = 30.0
		self.classification_rand_augment_max_shear_ratio = 0.3
		self.classification_rand_augment_max_translate_ratio = 0.45
		self.classification_rand_augment_max_enhance_delta = 0.9

		self.classification_mixup_alpha = 0.0
		self.classification_mixup_prob = 1.0
		self.classification_cutmix_alpha = 0.0
		self.classification_cutmix_prob = 1.0
		self.classification_mix_switch_prob = 0.5
		self.classification_label_smoothing = 0.0
		self.classification_metric_top_k = 5

		# Optional timm-like classification secondary augments.
		# Keep all defaults disabled so existing training behavior is unchanged.
		self.classification_use_color_jitter = False
		self.classification_color_jitter_prob = 1.0
		self.classification_color_jitter_random_order = True
		self.classification_color_jitter_brightness_delta = 32.0
		self.classification_color_jitter_contrast_range = (0.6, 1.4)
		self.classification_color_jitter_saturation_range = (0.6, 1.4)
		self.classification_color_jitter_hue_delta = 0.1

		self.classification_grayscale_prob = 0.0
		self.classification_gaussian_blur_prob = 0.0
		self.classification_gaussian_blur_kernel_size = 23
		self.classification_gaussian_blur_sigma_range = (0.1, 2.0)

		self.classification_random_erasing_prob = 0.0
		self.classification_random_erasing_use_fill_noise_color = False
		self.classification_random_erasing_min_area_size = 0.02
		self.classification_random_erasing_max_area_size = 0.25
		self.classification_random_erasing_min_area_count = 1
		self.classification_random_erasing_max_area_count = 1
		self.classification_random_erasing_fill_constant_color = [0, 0, 0]

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
		target_h = self.crop_height
		target_w = self.crop_width

		if not training:
			target_h = self.eval_crop_height if self.eval_crop_height is not None else self.crop_height
			target_w = self.eval_crop_width if self.eval_crop_width is not None else self.crop_width

		augments = []

		if training:
			augments.append(RandomResizedCropImageAugment(
				target_height=target_h,
				target_width=target_w,
				area_range=self.classification_crop_area_range,
				aspect_ratio_range=self.classification_crop_aspect_ratio_range,
				max_attempts=self.classification_crop_max_attempts,
			))
			augments.append(RandomFlipImageAugment(self.prob_of_flip))

			if self.classification_use_rand_augment:
				augments.append(RandAugmentImageAugment(
					num_layers=self.classification_rand_augment_num_layers,
					magnitude=self.classification_rand_augment_magnitude,
					magnitude_max=self.classification_rand_augment_magnitude_max,
					magnitude_std=self.classification_rand_augment_magnitude_std,
					op_execute_prob=self.classification_rand_augment_op_prob,
					max_rotate_degree=self.classification_rand_augment_max_rotate_degree,
					max_shear_ratio=self.classification_rand_augment_max_shear_ratio,
					max_translate_ratio=self.classification_rand_augment_max_translate_ratio,
					max_enhance_delta=self.classification_rand_augment_max_enhance_delta,
				))

			if self.classification_use_color_jitter:
				augments.append(RandomPhotoMetricDistortions(
					include_brightness=True,
					random_order=self.classification_color_jitter_random_order,
					execute_prob=self.classification_color_jitter_prob,
					brightness_max_delta=self.classification_color_jitter_brightness_delta,
					contrast_lower=self.classification_color_jitter_contrast_range[0],
					contrast_upper=self.classification_color_jitter_contrast_range[1],
					saturation_lower=self.classification_color_jitter_saturation_range[0],
					saturation_upper=self.classification_color_jitter_saturation_range[1],
					hue_max_delta=self.classification_color_jitter_hue_delta,
					brightness_prob=1.0,
					contrast_prob=1.0,
					saturation_prob=1.0,
					hue_prob=1.0,
				))
			else:
				# Keep the previous default path unchanged when color jitter is disabled.
				if self.random_brightness:
					augments.append(RandomBrightnessAugment(execute_prob=0.5))

			if self.classification_grayscale_prob > 0 + 1e-3:
				augments.append(RandomGrayscaleImageAugment(execute_prob=self.classification_grayscale_prob))

			if self.classification_gaussian_blur_prob > 0 + 1e-3:
				augments.append(RandomGaussianBlurImageAugment(
					kernel_size=self.classification_gaussian_blur_kernel_size,
					sigma_min=self.classification_gaussian_blur_sigma_range[0],
					sigma_max=self.classification_gaussian_blur_sigma_range[1],
					execute_prob=self.classification_gaussian_blur_prob,
				))

			if self.random_jepg_quality:
				augments.append(RandomJEPGQualityAugment())

			if self.classification_random_erasing_prob > 0 + 1e-3:
				augments.append(RandomErasingAugment(
					prob=self.classification_random_erasing_prob,
					min_area_size=self.classification_random_erasing_min_area_size,
					max_area_size=self.classification_random_erasing_max_area_size,
					min_area_count=self.classification_random_erasing_min_area_count,
					max_area_count=self.classification_random_erasing_max_area_count,
					fill_constant_color=self.classification_random_erasing_fill_constant_color,
					use_fill_noise_color=self.classification_random_erasing_use_fill_noise_color,
					ignore_label=self.ignore_label,
				))
		else:
			resize_short_edge = int(round(float(min(target_h, target_w)) / float(self.eval_crop_pct)))
			augments.append(ResizeShortEdgeImageAugment(short_edge=resize_short_edge))
			augments.append(CenterCropImageAugment(target_h, target_w))

			if self.random_noisy_eval_level > 0 + 1e-3:
				augments.append(RandomNoisyEvalAugment(noise_level=self.random_noisy_eval_level))

		pipeline_name = "imagenet1k_train_pipeline" if training else "imagenet1k_eval_pipeline"

		return ImageNet1KAugmentationsPipeLine(
			target_height=target_h,
			target_width=target_w,
			augments=augments,
			name=pipeline_name,
		)

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
