#!/usr/bin/python3

import numpy as np
import sys
import cntk as C

class AutoencoderVectorizer:
	def __init__(self):
		C.device.try_set_default_device(C.device.gpu(0))

		z = C.load_model('largedata/autoencoder_checkpoint')
		self._input_var = z.arguments[0]

		latent_mean = C.combine([z.find_by_name('latent_mean').owner])
		latent_log_sigma = C.combine([z.find_by_name('latent_log_sigma').owner])

		self._encoding_out = C.ops.splice(latent_mean, latent_log_sigma)


	def encode(self, img_data):
		return self._encoding_out.eval({self._input_var: img_data}).reshape(-1)


