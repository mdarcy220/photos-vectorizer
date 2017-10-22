
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import scipy
import scipy.ndimage
import re
import heapq

import cntk as C

C.device.try_set_default_device(C.device.gpu(0))
# Select the right target device when this is being tested:
if 'TEST_DEVICE' in os.environ:
	if os.environ['TEST_DEVICE'] == 'cpu':
		C.device.try_set_default_device(C.device.cpu())
	else:
		C.device.try_set_default_device(C.device.gpu(0))

# Ensure we always get the same amount of randomness
np.random.seed(0)

image_height = 150
image_width  = 150
num_channels = 3
image_dim = image_height * image_width * num_channels
image_shape = (num_channels, image_height, image_width)

x = C.input_variable(image_dim)
y = C.input_variable(image_dim)

encoded_size = 512

# Input variable and normalization
input_var = C.ops.input_variable(image_shape, np.float32)
scaled_input = C.ops.element_divide(input_var, C.ops.constant(256.), name="input_node")
noisy_scaled_input = C.ops.plus(scaled_input, C.random.normal(image_shape, scale=0.02))

cMap = 16
conv1	= C.layers.Convolution2D  ((3,3), 8, pad=True, activation=C.ops.tanh)(noisy_scaled_input)


conv2 = C.layers.Convolution2D ((3,3), cMap, pad=True, activation=C.ops.tanh)(C.layers.Dropout(0.05)(conv1))
pconv2  = C.layers.Convolution2D ((3,3), cMap, strides=(3,3), pad=True, activation=C.ops.tanh)(conv2)
conv3 = C.layers.Convolution2D ((5,5), cMap*2, strides=(2,2), pad=True, activation=C.ops.tanh)(C.layers.Dropout(0.1)(pconv2))
conv4 = C.layers.Convolution2D ((5,5), cMap, pad=True, strides=(2,2), activation=C.ops.tanh)(C.layers.Dropout(0.1)(conv3))

fc1 = C.layers.Dense(1024, activation=None)(conv4)
act1 = C.ops.tanh(fc1)
fc2 = C.layers.Dense(encoded_size, activation=None)(act1)
act2 = C.ops.tanh(fc2)

fc3 = C.layers.Dense(2048, activation=C.ops.leaky_relu)(C.ops.placeholder(shape=encoded_size))
fc5 = C.layers.Dense(16*50*50, activation=C.ops.leaky_relu)(fc3)

rs1 = C.ops.reshape(fc5, (16,50,50))
pdeconv1 = C.layers.ConvolutionTranspose2D((3,3), cMap, strides=(3,3), pad=False, bias=False, init=C.glorot_uniform(1), name="sln")(rs1)
deconv1 = C.layers.ConvolutionTranspose2D((5,5), cMap, pad=True, name="sln")(pdeconv1)

deconv2 = C.layers.ConvolutionTranspose2D((5,5), num_channels, activation=None, pad=True, bias=False, name="output_node")(deconv1)

latent_log_sigma = C.layers.Dense(encoded_size, activation=None)(act2)
latent_mean = C.layers.Dense(encoded_size, activation=C.ops.tanh)(act2)
latent_sigma = C.ops.exp(latent_log_sigma)
latent_vec = C.ops.plus(latent_mean, C.ops.element_times(latent_sigma, C.random.normal_like(latent_log_sigma)))
latent_kl_loss = -0.5 * C.ops.reduce_mean(1 + latent_log_sigma - C.ops.square(latent_mean) - latent_sigma, axis=-1)


z = deconv2(latent_vec)

decode_input = C.ops.input_variable(encoded_size)
decode_output = deconv2(decode_input)

err	  = C.ops.reshape(C.ops.minus(z, scaled_input), (image_dim))
sq_err	  = C.ops.square(err)
mse	  = C.ops.reduce_mean(sq_err)
rmse_loss = C.ops.sqrt(mse)
rmse_eval = rmse_loss

overall_loss = rmse_loss + latent_kl_loss

print('Net constructed. Loading images... ', end='') ; sys.stdout.flush()
input_img_filenames = ['../data/lfw/all/{}'.format(f) for f in os.listdir('../data/lfw/all') if re.match('.*\\.(png|jpg)',f)]
input_imgs_raw = [scipy.ndimage.io.imread(file)[50:-50,50:-50] for file in input_img_filenames[:4000]]
input_imgs_reshaped = np.array([np.ascontiguousarray(np.rollaxis(np.rollaxis(np.array(img, dtype=np.float32), 0, 3), 0, 3)) for img in input_imgs_raw])
print('Done.')


def test_image(img_num):
	img = np.clip(noisy_scaled_input.eval({input_var: input_imgs_reshaped[img_num]}).reshape(image_shape), 0.0, 255.0).astype(np.float32)
	tmp2 = C.ops.clip(z, 0.0, 1.0).eval({input_var: img})
	print("MSE Loss: {}".format(rmse_loss.eval({input_var: img})))
	output_img = np.rollaxis(np.rollaxis(tmp2.reshape(image_shape), 2, -3), 2, -3)
	plt.rcParams["figure.figsize"] = [12,9]
	plt.subplot(1, 3, 1)
	plt.imshow(np.rollaxis(np.rollaxis(img, 2, -3), 2, -3).astype(np.float64))
	plt.subplot(1, 3, 2)
	plt.imshow(output_img)
	plt.subplot(1, 3, 3)
	new_input = (np.array(np.random.normal(size=encoded_size), dtype=np.float32)+0.01)**(2)
	tmp3 = C.ops.clip(decode_output, 0.0, 1.0).eval({decode_input: new_input})[0][:3]
	output_img2 = np.rollaxis(np.rollaxis(tmp3.reshape(image_shape), 2, -3), 2, -3)
	plt.imshow(output_img2)
	plt.show()

test_image(10)

def encode(img_data):
	return latent_mean.eval({input_var: img_data})

# training config
max_epochs = 12
epoch_size = 2048
minibatch_size = 256

mn = 0

# Set learning parameters
lr_schedule = C.learning_rate_schedule([0.01], C.learners.UnitType.sample, epoch_size)
mm_schedule = C.learners.momentum_as_time_constant_schedule([1900], epoch_size)

# Instantiate the trainer object to drive the model training
#learner = C.learners.nesterov(z.parameters, lr_schedule, mm_schedule, unit_gain=True)
#learner = C.learners.adadelta(z.parameters)
learning_rate = 1.6e-3
lr_schedule = C.learning_rate_schedule([learning_rate * (0.95**i) for i in range(30)], C.UnitType.sample, epoch_size=12000)
beta1 = C.momentum_schedule(0.9)
beta2 = C.momentum_schedule(0.999)
learner = C.adam(z.parameters,
	lr=lr_schedule,
	momentum=beta1,
	variance_momentum=beta2,
	epsilon=1.5e-8,
	gradient_clipping_threshold_per_sample=3.0)
progress_printer = C.logging.ProgressPrinter(tag='Training')
trainer = C.Trainer(z, (overall_loss, rmse_eval), learner, progress_printer)


C.logging.log_number_of_parameters(z) ; print()

data = {input_var: input_imgs_reshaped};

# Get minibatches of images to train with and perform model training
for epoch in range(25):
	sample_count = 0
	while sample_count < epoch_size:  # loop over minibatches in the epoch
		trainer.train_minibatch({input_var: input_imgs_reshaped[np.random.randint(len(input_imgs_reshaped), size=minibatch_size)]})
		sample_count += minibatch_size

	trainer.summarize_training_progress()

img_lookup_table = {}
for i in range(len(input_imgs_reshaped)):
	img_lookup_table[i] = encode(input_imgs_reshaped[i]).reshape(-1)
	if i % 1000 == 0:
		print('Added a total of {} images to the table'.format(i))

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
		if i % (k_max + 500) == 0:
			min_dist_keys = sorted(min_dist_keys)[:k_max]
	return sorted(min_dist_keys)[:k_max]

test_image(10)

plt.subplot(3,3,2)
plt.imshow(input_imgs_raw[4])
plt.title('Original image')

lookup_result = lookup_img(input_imgs_reshaped[4], k_max=3)
for i in range(len(lookup_result)):
	result = lookup_result[i]
	plt.subplot(3,3,i+4)
	plt.imshow(input_imgs_raw[result[1]])
	plt.title('Result {}: diff={:.4f}'.format(i+1, result[0]))
	plt.axis('off')
lookup_result = lookup_img(input_imgs_reshaped[4], k_max=3, find_worst=True)
for i in range(len(lookup_result)):
	result = lookup_result[i]
	plt.subplot(3,3,i+7)
	plt.imshow(input_imgs_raw[result[1]])
	plt.title('Result {}: diff={:.4f}'.format(i+1, -1*result[0]))
	plt.axis('off')
plt.show()


