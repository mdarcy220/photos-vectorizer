#!/usr/bin/python3

import numpy as np

class FlatVectorizer:
	def __init__(self):
		pass

	def encode(self, img_data):
		return np.flatten(img_data)


