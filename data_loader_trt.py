import glob
import os

import cv2
import numpy as np


def preprocess(image_path, img_size=640):
    img0 = cv2.imread(image_path)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0
    if img.ndim == 3:
        img = np.expand_dims(img, 0)
    return img


def get_image_paths(image):
    print(f"Reading images from: {image}")
    paths = []
    if os.path.isdir(image):
        exts = {'*.jpg', '*.jpeg', '*.png'}
        for ext in exts:
            paths.extend(glob.glob(os.path.join(image, ext)))
    else:
        paths = [image]
    print(f"Found {len(paths)} images.")
    return paths


def load_data():
    # Yield 100-500 representative samples for good accuracy
    paths = get_image_paths('../furnas_dataset_v0.07/split_dataset/val/images')
    for path in paths:
        input_tensor = preprocess(path)
        # The key MUST match your model's input node name (e.g., 'input' or 'data')
        yield {"images": input_tensor}
