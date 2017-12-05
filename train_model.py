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


import version_check
version_check.assert_min_version('3.5')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import heapq
import cntk as C
import model_constructor
from ImageDataLoader import FilesystemImageDataLoader
import argparse

save_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'largedata', "autoencoder_checkpoint")
parser = argparse.ArgumentParser()
parser.add_argument('--no-plots', dest='no_plots', default=False, action='store_true', help='Disable pyplot displays before and after training')
parser.add_argument('--num-epochs', dest='num_epochs', default=sys.maxsize, action='store', type=int, help='Number of epochs to train for')
parser.add_argument('--save-interval', dest='save_interval', default=sys.maxsize, action='store', type=int, help='How often to save the network')
parser.add_argument('--display-interval', dest='display_interval', default=-1, action='store', type=int, help='How often (in minibatches) to display the training progress')
parser.add_argument('--minibatch-size', dest='minibatch_size', default=128, action='store', type=int, help='Minibatch size')
parser.add_argument('--start-lr', dest='start_lr', default=1.4e-3, action='store', type=float, help='Initial learning rate')
parser.add_argument('--lr-decay', dest='lr_decay', default=0.93, action='store', type=float, help='Learning rate decrease factor')
parser.add_argument('--save-filename', dest='save_filename', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'largedata', "autoencoder_checkpoint"), action='store', type=str, help='File to save the model in')
cmdargs = parser.parse_args(sys.argv[1:])

C.device.try_set_default_device(C.device.gpu(0))
# Select the right target device when this is being tested:
if 'TEST_DEVICE' in os.environ:
	if os.environ['TEST_DEVICE'] == 'cpu':
		C.device.try_set_default_device(C.device.cpu())
	else:
		C.device.try_set_default_device(C.device.gpu(0))

# Ensure we always get the same amount of randomness
np.random.seed(0)

image_height = 128
image_width  = 128
num_channels = 3
image_shape = (num_channels, image_height, image_width)
encoded_size = 1024

outputs, input_var, overall_loss = model_constructor.construct_model(image_height, image_width, num_channels, encoded_size=encoded_size)
model = outputs[0]
decode_output = outputs[1]
decode_input = decode_output.arguments[0]

latent_mean = C.combine([model.find_by_name('latent_mean').owner])
noisy_scaled_input = C.combine([model.find_by_name('noisy_scaled_input').owner])
rmse_loss = C.combine([overall_loss.find_by_name('rmse').owner])
rmse_eval = rmse_loss


print('Net constructed. Loading images... ', end='') ; sys.stdout.flush()
image_loader = FilesystemImageDataLoader(img_size=(image_width, image_height))
_, input_imgs_reshaped = image_loader.get_all_image_data(max_imgs=None)
print('Done.')


#plt.ion()
fig = plt.gcf()
fig.set_size_inches(12,9)
def test_image(img_num):

	img = np.clip(noisy_scaled_input.eval({input_var: input_imgs_reshaped[img_num]}).reshape(image_shape), 0.0, 1.0).astype(np.float32)
	tmp2 = C.ops.clip(model, 0.0, 1.0).eval({input_var: img})

	print("RMSE Loss: {}".format(rmse_loss.eval({input_var: img})))

	output_img = np.rollaxis(np.rollaxis(tmp2.reshape(image_shape), 2, -3), 2, -3)

	plt.subplot(1, 3, 1)
	plt.imshow(np.rollaxis(np.rollaxis(img, 2, -3), 2, -3).astype(np.float64))

	plt.subplot(1, 3, 2)
	plt.imshow(output_img)

	new_input = (np.array(np.random.normal(size=encoded_size), dtype=np.float32)+0.01)**(2)
	tmp3 = C.ops.clip(decode_output, 0.0, 1.0).eval({decode_input: new_input})[0][:3]
	output_img2 = np.rollaxis(np.rollaxis(tmp3.reshape(image_shape), 2, -3), 2, -3)

	plt.subplot(1, 3, 3)
	plt.imshow(output_img2)

	#plt.draw()
	#plt.pause(0.0000001)
	plt.show()

if not cmdargs.no_plots:
	test_image(10)


def init_trainer(epoch_size=32768, minibatch_size=128, start_lr=1.4e-3, lr_decay=0.93):
	# Set learning parameters
	#lr_schedule = C.learning_rate_schedule([0.01], C.learners.UnitType.sample, epoch_size)
	#mm_schedule = C.learners.momentum_as_time_constant_schedule([1900], epoch_size)

	# Instantiate the trainer object to drive the model training
	#learner = C.learners.nesterov(model.parameters, lr_schedule, mm_schedule, unit_gain=True)
	#learner = C.learners.adadelta(model.parameters)
	learning_rate = start_lr
	lr_schedule = C.learning_rate_schedule([learning_rate * (lr_decay**i) for i in range(30)], C.UnitType.sample, epoch_size=epoch_size)
	beta1 = C.momentum_schedule(0.9)
	beta2 = C.momentum_schedule(0.999)
	learner = C.adam(model.parameters,
		lr=lr_schedule,
		momentum=beta1,
		variance_momentum=beta2,
		epsilon=1.5e-8,
		gradient_clipping_threshold_per_sample=3.0)
	progress_printer = C.logging.ProgressPrinter(tag='Training', freq=(minibatch_size*cmdargs.display_interval if cmdargs.display_interval >= 0 else None))
	trainer = C.Trainer(model, (overall_loss, rmse_eval), learner, progress_printer)

	return trainer

C.logging.log_number_of_parameters(model) ; print()


minibatch_size = cmdargs.minibatch_size
epoch_size = 16384
trainer = init_trainer(epoch_size=epoch_size, minibatch_size=minibatch_size, start_lr=cmdargs.start_lr, lr_decay=cmdargs.lr_decay)

# Get minibatches of images to train with and perform model training
for epoch in range(cmdargs.num_epochs):
	sample_count = 0
	while sample_count < epoch_size:  # loop over minibatches in the epoch
		trainer.train_minibatch({input_var: input_imgs_reshaped[np.random.choice(len(input_imgs_reshaped), size=minibatch_size, replace=False)]})
		sample_count += minibatch_size
	if (epoch+1) % cmdargs.save_interval == 0:
		model.save(cmdargs.save_filename)
	trainer.summarize_training_progress()

model.save(cmdargs.save_filename)

if not cmdargs.no_plots:
	test_image(10)


