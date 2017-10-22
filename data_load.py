
import scipy
import scipy.ndimage
import re
import os
import numpy as np

default_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'largedata', 'train')

def get_images(data_dir=default_data_dir):
	input_img_filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if re.match('.*\\.(png|jpg)',f)]
	input_imgs_raw = [scipy.ndimage.io.imread(file)[50:-50,50:-50] for file in input_img_filenames[:8000]]
	input_imgs_reshaped = np.array([np.ascontiguousarray(np.rollaxis(np.rollaxis(np.array(img, dtype=np.float32), 0, 3), 0, 3)) for img in input_imgs_raw])
	return (input_imgs_raw, input_imgs_reshaped)
