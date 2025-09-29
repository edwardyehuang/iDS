import os
import numpy as np
import tensorflow as tf

import keras

import iseg.static_strings as ss
import tqdm

from PIL import Image


from iseg.data_process.augments import RandomCropAugment, RandomRotateAugment
from iseg.utils.common import get_tensor_shape

IMAGE_DIR = "images"
LABEL_DIR = "gt/semantic/label_images"

IMAGE_FILE_EXTENSION = ".jpg"
LABEL_FILE_EXTENSION = ".png"

VAL_START_ID = 476

SEMANTIC_DRONE_VAL_PATCH_COUNT = -1

ORGINAL_VAL_IMAGE_COUNT = 80

from ids.dataset import Dataset


def create_val_imagesets(
    image_dir, output_path,
):
    
    image_names = os.listdir(image_dir)

    with open(output_path, "w") as f:
        for image_name in image_names:
            image_name_wo_ext = os.path.splitext(image_name)[0]
            image_id = int(image_name_wo_ext)

            if image_id >= VAL_START_ID:
                f.write(f"{image_id}\n")


def get_colormap():

    return np.array(
        [
            (0, 0, 0),
            (128, 64, 128),
            (130, 76, 0),
            (0, 102, 0),
            (112, 103, 87),
            (28, 42, 168),
            (48, 41, 30),
            (0, 50, 89),
            (107, 142, 35),
            (70, 70, 70),
            (102, 102, 156),
            (254, 228, 12),
            (254, 148, 12),
            (190, 153, 153),
            (153, 153, 153),
            (255, 22, 96),
            (102, 51, 0),
            (9, 143, 150),
            (119, 11, 32),
            (51, 51, 0),
            (190, 250, 190),
            (112, 150, 146),
            (2, 135, 115),
            (255, 0, 0)
        ], 
        np.uint8
    )



class SemanticDrone(Dataset):
    def __init__(self, dataset_dir):

        super().__init__(dataset_dir)

        self.ignore_label = 0
        self.num_class = 23
        self.val_image_count = 6000
        self.compress = True


        self.pil_palette = []

        for color in get_colormap():
            self.pil_palette.extend(color)

        self.pil_palette += [0] * (768 - len(self.pil_palette))


    def load_data_paths(self, dataset_dir):

        image_dir = os.path.join(dataset_dir, IMAGE_DIR)
        label_dir = os.path.join(dataset_dir, LABEL_DIR)

        image_name_list = os.listdir(image_dir)

        train_image_path = []
        train_label_path = []
        
        val_image_path = []
        val_label_path = []

        for image_name in image_name_list:

            # get name w/o ext
            image_name_wo_ext = os.path.splitext(image_name)[0]
            image_id = int(image_name_wo_ext)

            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, image_name_wo_ext + LABEL_FILE_EXTENSION)

            if not os.path.exists(label_path):
                raise ValueError(f"Label path {label_path} not exists!")
            

            if image_id < VAL_START_ID:
                train_image_path.append(image_path)
                train_label_path.append(label_path)
            else:
                val_image_path.append(image_path)
                val_label_path.append(label_path)


        if len(val_image_path) != ORGINAL_VAL_IMAGE_COUNT:
            raise ValueError(f"Val image count should be {ORGINAL_VAL_IMAGE_COUNT}, but got {len(val_image_path)}")


        return ((train_image_path, train_label_path), (val_image_path, val_label_path))
    

    def load_trainval_tensor_ds(self):
        train_ds, val_ds = super().load_trainval_tensor_ds()

        if val_ds is not None:
            val_ds = self.build_val_data_crop_patches(val_ds)

        return train_ds, val_ds


    def build_val_data_crop_patches(self, ds):

        global SEMANTIC_DRONE_VAL_PATCH_COUNT

        patches_ds = ds.flat_map(self.split_data_into_crop_patches)

        if SEMANTIC_DRONE_VAL_PATCH_COUNT < 0:
            # compute number of patches (quickly loop), tqdm
            num_patches = sum(1 for _ in tqdm.tqdm(patches_ds))

            SEMANTIC_DRONE_VAL_PATCH_COUNT = num_patches

        self.val_image_count = SEMANTIC_DRONE_VAL_PATCH_COUNT

        return patches_ds


    @tf.autograph.experimental.do_not_convert
    def split_data_into_crop_patches(self, image, label):
        """
        Split each (image, label) pair into fixed-size patches.
        If the image dimensions are not divisible by crop size, the last
        patch along that dimension is shifted so the border is covered
        (causing overlap).
        """

        image_patches = tf.image.extract_patches(
            images=tf.expand_dims(image, axis=0),  # [1, H, W, C]
            sizes=[1, self.crop_height, self.crop_width, 1],
            strides=[1, self.crop_height, self.crop_width, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        label_patches = tf.image.extract_patches(
            images=tf.expand_dims(label, axis=0),  # [1, H, W, 1]
            sizes=[1, self.crop_height, self.crop_width, 1],
            strides=[1, self.crop_height, self.crop_width, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        image_patches = tf.reshape(image_patches, [-1, self.crop_height, self.crop_width, image.shape[-1]])
        label_patches = tf.reshape(label_patches, [-1, self.crop_height, self.crop_width, label.shape[-1]])

        ds = tf.data.Dataset.from_tensor_slices((image_patches, label_patches))

        return ds
    

    def create_augment_pipeline(self, training):

        pipeline = super().create_augment_pipeline(training=training)

        # for train set, we need to crop it first to reduce the computational burden
        if training:

            larger_crop_height = int(self.crop_height / self.min_scale_factor)
            larger_crop_width = int(self.crop_width / self.min_scale_factor)

            pipeline.augments.insert(0, RandomCropAugment(larger_crop_height, larger_crop_width))
            pipeline.augments.insert(1, RandomRotateAugment(
                prob_of_rotate=0.5,
                fill_constant_color=[0, 0, 0],
                ignore_label=self.ignore_label,
            ))

        return pipeline
    


    def load_tensor_from_path(self, image_path, label_path):
        
        image_data = tf.io.read_file(image_path)

        try:
            image_tensor = tf.image.decode_jpeg(
                image_data, 
                channels=3, 
                dct_method="INTEGER_ACCURATE", 
                try_recover_truncated=True
            )

            image_tensor = tf.cast(image_tensor, tf.float32)

            label_tensor = None

            if label_path is not None:
                label_tensor = self.load_rgb_label_to_tensor(label_path)

            return image_tensor, label_tensor

        except:

            if label_path is not None:
                raise ValueError(f"Error: {image_path} or {label_path} is not a valid image.")
            else:
                raise ValueError(f"Error: {image_path} is not a valid JPEG image.")
            

    
    def load_rgb_label_to_tensor(self, label_path):

        tensor = tf.py_function(self._load_label_to_tensor_internel, [label_path], tf.int32)
        tensor.set_shape([None, None, 1])

        return tensor
    

    def _load_label_to_tensor_internel(self, path_tensor):

        if isinstance(path_tensor, str):
            label_path = path_tensor
        else:
            label_path = path_tensor.numpy()

        label_image = Image.open(label_path)

        pal_image = Image.new('P', (1, 1))
        pal_image.putpalette(self.pil_palette)

        label_image = label_image.quantize(len(get_colormap()), palette=pal_image)

        label_array = keras.preprocessing.image.img_to_array(label_image, "channels_last")

        label_image.close()
        pal_image.close()

        label_tensor = tf.convert_to_tensor(label_array)
        label_tensor = tf.cast(label_tensor, tf.int32)

        return label_tensor