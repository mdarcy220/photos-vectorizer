
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

image_height = 128
image_width  = 128
num_channels = 3
encoded_size = 512

z = C.load_model('largedata/autoencoder_checkpoint')
input_var = z.arguments[0]

latent_mean = C.combine([z.find_by_name('latent_mean').owner])
latent_log_sigma = C.combine([z.find_by_name('latent_log_sigma').owner])
latent_sigma = C.combine([z.find_by_name('latent_sigma').owner])
noisy_scaled_input = C.combine([z.find_by_name('noisy_scaled_input').owner])

print('Net loaded. Loading images... ', end='') ; sys.stdout.flush()
image_loader = data_load.ImageDataLoader()
print('Done.')

encoding_out = C.ops.splice(latent_mean, latent_log_sigma)

def encode(img_data):
	return encoding_out.eval({input_var: img_data})

img_lookup_table = {}
#for i in range(image_loader.num_images()):
for i in range(10):
	img_lookup_table[i] = encode(image_loader.get_reshaped(i)).reshape(-1)
	if i % 1000 == 0:
		print('Added a total of {} images to the table'.format(i), end="\r")
print()

# Free up some memory
#input_imgs_reshaped = None

def lookup_img(img_data, k_max=1, find_worst=False):
	encoded_image = encode(img_data).reshape(-1)
	min_dist_keys = []
	i = 0
	for key in img_lookup_table:
		diff = encoded_image - img_lookup_table[key]
		#print(img_lookup_table[key])
		dist = np.dot(diff, diff)

		# If we're looking for the worst match, reverse the distance
		if find_worst:
			dist *= -1
		min_dist_keys.append((dist, key))

		# Save memory by periodically cleaning cruft
		if i % (k_max + 500) == 0:
			min_dist_keys = sorted(min_dist_keys)[:k_max]
	return sorted(min_dist_keys)[:k_max]


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

if __name__ == '__main__':
	try:
		while True:
			imgnum = int(input('Enter an image number (0-{}): '.format(len(img_lookup_table)-1)))
			display_lookup_best_and_worst(image_loader.get_reshaped(imgnum))
	except EOFError:
		print('exit')

