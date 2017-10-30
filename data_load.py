
import scipy
import scipy.ndimage
import re
import os
import numpy as np
import random
import skimage.transform
import warnings

default_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'largedata', 'train')

## Loads images from the data directory.
#
# @param data_dir (str)
# <br>	The directory to look for images in
#
# @param max_imgs (int or None)
# <br>	The maximum number of images to retrieve
#
# @param random_order (bool)
# <br>	Whether to load random images found. This is useful when max_imgs is
# set, as it allows getting randomly sampled images. For example, if
# max_imgs=100 and there are 500 images in data_dir, the method will return 100
# random images from data_dir instead of just taking the first 100
#
def get_images(data_dir=default_data_dir, max_imgs=None, random_order=False, img_size=(128,128)):
	input_img_filenames = []
	for root, subd, files in os.walk(data_dir):
		input_img_filenames += [os.path.join(root, f) for f in files if re.match('.*\\.(png|jpg)',f)]

	if random_order:
		random.shuffle(input_img_filenames)
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		input_imgs_raw = [skimage.transform.resize(scipy.ndimage.io.imread(file), img_size) for file in input_img_filenames[:max_imgs]]
	input_imgs_reshaped = np.array([np.ascontiguousarray(np.rollaxis(np.rollaxis(np.array(img, dtype=np.float32), 0, 3), 0, 3)) for img in input_imgs_raw])
	print(np.max(input_imgs_raw[0]))
	return (input_imgs_raw, input_imgs_reshaped)
