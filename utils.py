import math
import cv2
import shutil
import os
from PIL import Image


# Source: https://rubikscode.net/2021/11/15/receptive-field-arithmetic-for-convolutional-neural-networks/
class ReceptiveFieldCalculator():
    def calculate(self, architecture, input_image_size):
        input_layer = ('input_layer', input_image_size, 1, 1, 0.5)
        self._print_layer_info(input_layer)

        for key in architecture:
            current_layer = self._calculate_layer_info(architecture[key], input_layer, key)
            self._print_layer_info(current_layer)
            input_layer = current_layer

    def _print_layer_info(self, layer):
        print(f'------')
        print(f'{layer[0]}: n = {layer[1]}; r = {layer[3]}; j = {layer[2]}; start = {layer[4]}')
        print(f'------')

    def _calculate_layer_info(self, current_layer, input_layer, layer_name):
        n_in = input_layer[1]
        j_in = input_layer[2]
        r_in = input_layer[3]
        start_in = input_layer[4]

        k = current_layer[0]
        s = current_layer[1]
        p = current_layer[2]

        n_out = math.floor((n_in - k + 2 * p) / s) + 1
        padding = (n_out - 1) * s - n_in + k
        p_right = math.ceil(padding / 2)
        p_left = math.floor(padding / 2)

        j_out = j_in * s
        r_out = r_in + (k - 1) * j_in
        start_out = start_in + ((k - 1) / 2 - p_left) * j_in
        return layer_name, n_out, j_out, r_out, start_out

# Source: https://github.com/datable-be/grayscale-image-detector/blob/main/grayimage_detector.py
def is_quasi_monochrome_with_rgb(image_path):
    # Open image file
    image = Image.open(image_path)

    # Check if the image has an RGB channel
    bands = image.getbands()
    if len(bands) != 3 or 'R' not in bands or 'G' not in bands or 'B' not in bands:
        return False

    # Check if the image contains a black and white photo
    num_pixels = 0
    if image.mode == 'RGBA':
        # Split image into separate channels
        r, g, b, a = image.split()

        # Check if the alpha channel is mostly white
        alpha_pixels = a.getdata()
        num_white_pixels = sum(1 for pixel in alpha_pixels if pixel == 255)
        num_pixels = len(alpha_pixels)
        if num_white_pixels / num_pixels > 0.99:
            return True

    # Check if the image is monochrome
    if image.mode == 'L':
        return True

    # Check if the image is RGB but with only one color channel
    if image.mode == 'RGB':
        r, g, b = image.split()
        if r.getextrema() == g.getextrema() == b.getextrema():
            return True
        # Check if the image has very low color saturation
        hsv_image = image.convert('HSV')
        h, s, v = hsv_image.split()
        s_pixels = s.getdata()
        if num_pixels == 0:
            num_pixels = len(s_pixels)
        num_low_sat_pixels = sum(1 for pixel in s_pixels if pixel < 32)
        if num_low_sat_pixels / num_pixels > 0.95:
            return True

    return False


def cleanup_images(main_dir='./dataset/train/', gray_dir='./dataset/gray/', check=lambda image: False):
    images = os.listdir(main_dir)
    for i, image in enumerate(images):
        if check(main_dir + image):
            shutil.move(main_dir + image, gray_dir + image)
        if i % 100 == 0:
            print(f'{i + 1} / {len(images)}')

import os
import hashlib

def calculate_file_hash(file_path):
    """Calculate the SHA-256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def remove_duplicate_files(directory):
    """Remove duplicate files in a directory by comparing content."""
    hashes = {}  # Dictionary to store file hashes
    duplicates = []  # List to store duplicate file paths

    # Traverse the directory
    for root, _, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_hash = calculate_file_hash(file_path)

            if file_hash in hashes:
                # File is a duplicate if the hash already exists
                duplicates.append(file_path)
                print(f"Duplicate found: {file_path}")
            else:
                # Store the hash for unique files
                hashes[file_hash] = file_path

    # Remove duplicate files
    for duplicate_file in duplicates:
        os.remove(duplicate_file)
        print(f"Removed duplicate: {duplicate_file}")

from torchvision import transforms

def resize_images(input_directory='./dataset/train/flicker', output_directory='./dataset/train/flicker-resized'):
    # Define the transformation to resize the image
    transform = transforms.Compose([
        transforms.Resize((256, 256))
    ])

    # Walk through all subdirectories in the input directory
    for root, _, files in os.walk(input_directory):
        # Determine the relative path to keep the directory structure
        relative_path = os.path.relpath(root, input_directory)
        output_subdir = os.path.join(output_directory, relative_path)

        # Ensure the output subdirectory exists
        os.makedirs(output_subdir, exist_ok=True)

        # Process each file in the current directory
        for i, filename in enumerate(files):
            if filename.lower().endswith(".png") or filename.lower().endswith(".jpg"):
                # Construct full file paths for input and output
                input_path = os.path.join(root, filename)
                output_path = os.path.join(output_subdir, filename)

                # Open the image with PIL
                img = Image.open(input_path)

                # Apply the resizing transformation
                resized_img = transform(img)

                # Save the resized image to the output directory, preserving the structure
                resized_img.save(output_path)

                print(f"Resized and saved {output_path}")
            if i % 100 == 0:
                print(f'{i + 1} / {len(files)}')

# Flicker dataset: https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL

if __name__ == '__main__':
    pass
    #cleanup_images(gray_dir='./dataset/quasi_monochrome/', check=is_quasi_monochrome_with_rgb)
    #remove_duplicate_files(directory='./dataset/train')

    #cleanup_images(main_dir='./dataset/train/flicker/', gray_dir='./dataset/quasi_monochrome/', check=is_quasi_monochrome_with_rgb)
    #remove_duplicate_files(directory='./dataset/train')
    #resize_images(input_directory='./dataset/train/flicker', output_directory='./dataset/train/flicker-resized')
    #resize_images(input_directory='./dataset/val/flicker', output_directory='./dataset/val/flicker-resized')
    #resize_images(input_directory='./dataset/train/human', output_directory='./dataset/train/human-resized')
    #remove_duplicate_files(directory='./dataset/train/faces')