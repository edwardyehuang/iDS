import os
from absl import app, flags
from PIL import Image
import glob
from tqdm import tqdm

def blend_images(image_dir, label_dir, output_dir, alpha):
    """
    Blends images with their corresponding labels.

    Args:
        image_dir (str): Path to the directory containing images.
        label_dir (str): Path to the directory containing labels.
        output_dir (str): Path to the directory to save blended images.
        alpha (float): The alpha value for blending.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    label_paths = glob.glob(os.path.join(label_dir, '*.png'))

    print(f"Found {len(label_paths)} labels in {label_dir}")

    for label_path in tqdm(label_paths, desc="Blending images"):
        label_filename = os.path.basename(label_path)
        image_name = os.path.splitext(label_filename)[0]
        
        image_path = None
        for ext in ('.jpg', '.jpeg', '.png'):
            potential_image_path = os.path.join(image_dir, image_name + ext)
            if os.path.exists(potential_image_path):
                image_path = potential_image_path
                break
        
        if image_path is None:
            tqdm.write(f"Warning: Image for label {label_filename} not found in {image_dir}. Skipping.")
            continue
        
        image_filename = os.path.basename(image_path)

        try:
            image = Image.open(image_path).convert('RGB')
            label = Image.open(label_path)

            # Ensure label is in P mode and convert to RGB for blending
            if label.mode != 'P':
                print(f"Warning: Label {label_filename} is not in 'P' mode. It is in {label.mode}. Trying to convert.")
            
            # Convert label to RGB using its palette
            label_rgb = label.convert('RGB')

            if image.size != label_rgb.size:
                print(f"Warning: Size mismatch between {image_filename} ({image.size}) and {label_filename} ({label_rgb.size}). Skipping.")
                continue

            blended_image = Image.blend(image, label_rgb, alpha)
            
            output_path = os.path.join(output_dir, image_filename)
            blended_image.save(output_path)

        except Exception as e:
            tqdm.write(f"Error processing {image_filename}: {e}")

FLAGS = flags.FLAGS
flags.DEFINE_string('image_dir', None, 'Directory containing the original images.')
flags.DEFINE_string('label_dir', None, 'Directory containing the segmentation labels (P mode PNGs).')
flags.DEFINE_string('output_dir', None, 'Directory to save the blended images.')
flags.DEFINE_float('alpha', 0.5, 'Alpha for blending. 0.0 means only image, 1.0 means only label.')

flags.mark_flag_as_required('image_dir')
flags.mark_flag_as_required('label_dir')
flags.mark_flag_as_required('output_dir')

def main(_):
    blend_images(FLAGS.image_dir, FLAGS.label_dir, FLAGS.output_dir, FLAGS.alpha)

if __name__ == '__main__':
    app.run(main)
