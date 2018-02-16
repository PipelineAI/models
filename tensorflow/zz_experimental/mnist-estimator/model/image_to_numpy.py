#! /usr/env/bin python3

import skimage.io
import numpy as np
import json

def image_to_numpy(image_path):
    """Convert image to np array.
    Returns:
        Numpy array of size (1, 28, 28, 1) with MNIST sample images.
    """
    image_as_numpy = np.zeros((1, 28, 28))
    print(image_as_numpy.shape)
    for idx, image_path in enumerate(image_path):
        image_read = skimage.io.imread(fname=image_path, as_grey=True)
        print(image_read.shape)
        image_as_numpy[idx, :, :] = image_read
 
    return json.dumps(list(image_as_numpy.flatten()))

if __name__ == '__main__':
    print(image_to_numpy(['data/6.jpg']))
