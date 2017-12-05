#!/usr/bin/python3

# This file is part of Photos-vectorizer.
#
# Copyright (C) 2017  Mike D'Arcy
#
# Photos-vectorizer is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Photos-vectorizer is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License
# for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


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


