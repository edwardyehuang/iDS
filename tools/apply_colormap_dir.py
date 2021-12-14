# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import os, sys

rootpath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

sys.path.insert(1, rootpath)

import tensorflow as tf

import numpy as np
from PIL import Image

from absl import app

from absl import flags
from common_flags import FLAGS

from ids.voc2012 import get_colormap as get_voc2012_colormap
from ids.cityscapes_fine import get_colormap as get_cityscapes_colormap

flags.DEFINE_string("input_dir", None, "input dir path")
flags.DEFINE_string("output_dir", None, "output dir path")
flags.DEFINE_string("colormap", "voc2012", "colormap name")
flags.DEFINE_integer("ignore_label", 255, "ignore label")


def apply_colormap_to_dir(input_dir, output_dir=None, colormap=None):

    colormap = colormap.astype(np.uint8)

    counter = 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for filename in tf.io.gfile.listdir(input_dir):

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        img = Image.open(input_path)

        if img.mode != "L" and img.mode != "P":
            continue

        img = img.convert("P")

        img.putpalette(colormap)
        img.save(output_path, format="PNG")

        counter += 1

        tf.print("Processed {}".format(counter))


def main(argv):

    colormap_name = FLAGS.colormap

    colormap_name = colormap_name.lower()

    if colormap_name == "voc2012":
        colormap = get_voc2012_colormap()
    elif colormap_name == "cityscapes":
        colormap = get_cityscapes_colormap()
    else:
        raise ValueError(f"Not support colormap = {colormap_name}")

    if FLAGS.ignore_label == 0:
        colormap = colormap[1:]

    apply_colormap_to_dir(FLAGS.input_dir, FLAGS.output_dir, colormap=colormap)


if __name__ == "__main__":

    app.run(main)
