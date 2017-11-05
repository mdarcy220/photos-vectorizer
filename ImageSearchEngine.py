#!/usr/bin/python3

import numpy as np
import os
import sys
import time
import heapq
from ImageDataLoader import FilesystemImageDataLoader
import ImageVectorize


class ImageSearchEngine:
	def __init__(self, encoder_class=ImageVectorize.FlatVectorizer, max_images=sys.maxsize):
		self._max_images = max_images
		print('Initializing vectorizer... ', end='') ; sys.stdout.flush()
		self._vectorizer = encoder_class()
		print('Done.')

		self.image_loader = FilesystemImageDataLoader()

		# This must be called only AFTER initializing self._vectorizer
		self._img_lookup_table = self._init_lookup_table()


	def _init_lookup_table(self):
		status_str = 'Added a total of {:d} images to the lookup table'
		num_images = min(self.image_loader.num_images(), self._max_images)

		img_lookup_table = {}
		for i in range(num_images):
			img_lookup_table[i] = self._vectorizer.encode(self.image_loader.get_reshaped(i))
			if i % 500 == 0:
				print(status_str.format(i), end="\r")

		print(status_str.format(num_images))

		return img_lookup_table


	def lookup_img(self, img_data, k_max=1, find_worst=False):
		encoded_image = self._vectorizer.encode(img_data).reshape(-1)
		min_dist_keys = []
		i = 0
		for key in self._img_lookup_table:
			diff = encoded_image - self._img_lookup_table[key]
			dist = np.dot(diff, diff)

			# If we're looking for the worst match, reverse the distance
			if find_worst:
				dist *= -1
			min_dist_keys.append((dist, key))

			# Save memory by periodically cleaning cruft
			if i % (k_max + 500) == 0:
				min_dist_keys = sorted(min_dist_keys)[:k_max]
		return sorted(min_dist_keys)[:k_max]


# When run as a script, provide an interactive console and use pyplot to show
# results
if __name__ == '__main__':
	import matplotlib.pyplot as plt

	search_engine = ImageSearchEngine()

	def display_lookup_best_and_worst(image_to_lookup):
		plt.figure(figsize=(8,9))
		plt.subplot(3,3,2)
		plt.imshow(np.rollaxis(np.rollaxis(image_to_lookup, 0, 3), 2, 3))
		plt.title('Original image', fontsize=10)
		plt.gcf().subplots_adjust(hspace=0.5)

		lookup_result = lookup_img(image_to_lookup, k_max=3)
		for i in range(len(lookup_result)):
			result = lookup_result[i]
			plt.subplot(3,3,i+4)
			plt.imshow(image_loader.get_raw(result[1]))
			plt.title('Best {}: diff={:.3f}, id={}'.format(i+1, result[0], result[1]), fontsize=10)
			plt.axis('off')
		lookup_result = lookup_img(image_to_lookup, k_max=3, find_worst=True)
		for i in range(len(lookup_result)):
			result = lookup_result[i]
			plt.subplot(3,3,i+7)
			plt.imshow(image_loader.get_raw(result[1]))
			plt.title('Worst {}: diff={:.3f}, id={}'.format(i+1, -1*result[0], result[1]), fontsize=10)
			plt.axis('off')
		plt.show()

	try:
		while True:
			imgnum = int(input('Enter an image number (0-{}): '.format(len(img_lookup_table)-1)))
			display_lookup_best_and_worst(search_engine.image_loader.get_reshaped(imgnum))
	except EOFError:
		print('exit')

