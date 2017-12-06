
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
import os
import cntk as C
import skimage

class ResnetVectorizer:
	def __init__(self):
		C.device.try_set_default_device(C.device.gpu(0))

		raw_model = C.load_model(os.path.join(os.path.dirname(__file__), '..', 'saved_model.dnn'))
		z = C.combine(raw_model.find_by_name('z'))
		t0 = C.combine(z.outputs[0].owner.inputs[0].owner.inputs[0])
		self._input_var = t0.arguments[0]

		self._encoding_out = t0


	def encode(self, img_data):
		# Must reshape image to fit Resnet
		new_img_data = np.ascontiguousarray(np.rollaxis(np.rollaxis(np.array(img_data, dtype=np.float32), 0, -3), 0, 3))
		new_img_data = skimage.transform.resize(new_img_data[:,:,:3], (224,224), mode='reflect')
		new_img_data = np.ascontiguousarray(np.rollaxis(np.rollaxis(np.array(new_img_data, dtype=np.float32), 0, 3), 0, 3))
		return np.squeeze(self._encoding_out.eval({self._input_var: new_img_data}))



