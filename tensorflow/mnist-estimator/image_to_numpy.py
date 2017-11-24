#! /usr/env/bin python3

import skimage.io
import numpy as np

def image_to_numpy(image_path):
    """Convert image to np array.
    Returns:
        Numpy array of size (1, 28, 28, 1) with MNIST sample images.
    """
    images = np.zeros((10, 28, 28, 3))
    print(images.shape)
    for idx, image_path in enumerate(image_path):
        image_read = skimage.io.imread(image_path)
        print(image_read)
        images[idx, :, :, :] = image_read 
    return images

if __name__ == '__main__':
    print(image_to_numpy(['data/6.jpg']))
