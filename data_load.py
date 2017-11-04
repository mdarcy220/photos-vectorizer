
import scipy
import scipy.ndimage
import re
import os
import numpy as np
import random
import skimage.transform
import warnings
import sys

default_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'largedata', 'train')

class ImageDataLoader:

	## Constructor.
	#
	# @param data_dir (str)
	# <br>	The directory to look for images in
	#
	def __init__(self, data_dir=default_data_dir, random_order=False, max_cached_images=sys.maxsize, img_size=(128, 128)):
		self._data_dir = data_dir
		self._cache = {}
		self._img_size = img_size

		self._img_filenames = []
		for root, subd, files in os.walk(self._data_dir):
			self._img_filenames += [os.path.join(root, f) for f in files if re.match('.*\\.(png|jpg)',f)]
		if random_order:
			random.shuffle(self._img_filenames)


	## Loads all images from the data directory.
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
	# @return (tuple)
	# <br>	Format: (imgs_raw, imgs_reshaped)
	# <br>	Where `imgs_raw` is a list of Numpy arrays containing image
	# 	data, shaped however Scipy loaded them. The `imgs_reshaped`
	# 	field also contains a list of images (in the same order as
	# 	`imgs_raw`, but each image has the shape `(3, img_size[0],
	# 	img_size[1])`.
	def get_all_image_data(self, max_imgs=None):


		if max_imgs is None:
			max_imgs = sys.maxsize
		max_imgs = min(max_imgs, len(self._img_filenames))
		input_imgs_raw = [self.get_raw(img_num) for img_num in range(max_imgs)]
		input_imgs_reshaped = np.array([self.reshape_img(img) for img in input_imgs_raw])
		return (input_imgs_raw, input_imgs_reshaped)


	def get_raw(self, img_num):
		filename = self._img_filenames[img_num]
		return self.fix_img_size(scipy.ndimage.io.imread(filename))


	def get_reshaped(self, img_num):
		return self.reshape_img(self.get_raw(img_num))


	def reshape_img(self, img_data):
		return np.ascontiguousarray(np.rollaxis(np.rollaxis(np.array(img_data, dtype=np.float32), 0, 3), 0, 3))


	def fix_img_size(self, img_data):
		return skimage.transform.resize(img_data[:,:,:3], self._img_size, mode='reflect')


	def num_images(self):
		return len(self._img_filenames)
