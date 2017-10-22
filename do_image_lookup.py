
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import heapq
import cntk as C
import model_constructor
import data_load

C.device.try_set_default_device(C.device.gpu(0))

image_height = 150
image_width  = 150
num_channels = 3
encoded_size = 512

z = C.load_model('largedata/autoencoder_checkpoint')
input_var = z.arguments[0]

latent_mean = C.combine([z.find_by_name('latent_mean').owner])
noisy_scaled_input = C.combine([z.find_by_name('noisy_scaled_input').owner])

print('Net loaded. Loading images... ', end='') ; sys.stdout.flush()
input_imgs_raw, input_imgs_reshaped = data_load.get_images()
print('Done.')

image_to_lookup = input_imgs_reshaped[4]

def encode(img_data):
	return latent_mean.eval({input_var: img_data})

img_lookup_table = {}
for i in range(len(input_imgs_reshaped)):
	img_lookup_table[i] = encode(input_imgs_reshaped[i]).reshape(-1)
	if i % 1000 == 0:
		print('Added a total of {} images to the table'.format(i))

# Free up some memory
input_imgs_reshaped = None

def lookup_img(img_data, k_max=1, find_worst=False):
	encoded_image = encode(img_data).reshape(-1)
	min_dist_keys = []
	i = 0
	for key in img_lookup_table:
		diff = encoded_image - img_lookup_table[key]
		dist = np.dot(diff, diff)

		# If we're looking for the worst match, reverse the distance
		if find_worst:
			dist *= -1
		min_dist_keys.append((dist, key))

		# Save memory by periodically cleaning cruft
		if i % (k_max + 500) == 0:
			min_dist_keys = sorted(min_dist_keys)[:k_max]
	return sorted(min_dist_keys)[:k_max]


plt.subplot(3,3,2)
plt.imshow(np.rollaxis(np.rollaxis(image_to_lookup, 0, 3), 2, 3)/256)
plt.title('Original image')

lookup_result = lookup_img(image_to_lookup, k_max=3)
for i in range(len(lookup_result)):
	result = lookup_result[i]
	plt.subplot(3,3,i+4)
	plt.imshow(input_imgs_raw[result[1]])
	plt.title('Result {}: diff={:.4f}'.format(i+1, result[0]))
	plt.axis('off')
lookup_result = lookup_img(image_to_lookup, k_max=3, find_worst=True)
for i in range(len(lookup_result)):
	result = lookup_result[i]
	plt.subplot(3,3,i+7)
	plt.imshow(input_imgs_raw[result[1]])
	plt.title('Result {}: diff={:.4f}'.format(i+1, -1*result[0]))
	plt.axis('off')
plt.show()

