
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
image_shape = (num_channels, image_height, image_width)
encoded_size = 2048

outputs, input_var, overall_loss = model_constructor.construct_model(image_height, image_width, num_channels, encoded_size=encoded_size)
model = outputs[0]
decode_output = outputs[1]
decode_input = decode_output.arguments[0]

latent_mean = C.combine([model.find_by_name('latent_mean').owner])
noisy_scaled_input = C.combine([model.find_by_name('noisy_scaled_input').owner])
rmse_loss = C.combine([overall_loss.find_by_name('rmse').owner])
rmse_eval = rmse_loss


print('Net constructed. Loading images... ', end='') ; sys.stdout.flush()
_, input_imgs_reshaped = data_load.get_images()
print('Done.')


plt.ion()
fig = plt.gcf()
fig.set_size_inches(12,9)
def test_image(img_num):

	img = np.clip(noisy_scaled_input.eval({input_var: input_imgs_reshaped[img_num]}).reshape(image_shape), 0.0, 255.0).astype(np.float32)
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

	plt.draw()
	plt.pause(0.0000001)

test_image(10)

# training config
max_epochs = 12
epoch_size = 2048
minibatch_size = 128

mn = 0

# Set learning parameters
lr_schedule = C.learning_rate_schedule([0.01], C.learners.UnitType.sample, epoch_size)
mm_schedule = C.learners.momentum_as_time_constant_schedule([1900], epoch_size)

# Instantiate the trainer object to drive the model training
#learner = C.learners.nesterov(model.parameters, lr_schedule, mm_schedule, unit_gain=True)
#learner = C.learners.adadelta(model.parameters)
learning_rate = 1.4e-3
lr_schedule = C.learning_rate_schedule([learning_rate * (0.93**i) for i in range(30)], C.UnitType.sample, epoch_size=20000)
beta1 = C.momentum_schedule(0.9)
beta2 = C.momentum_schedule(0.999)
learner = C.adam(model.parameters,
	lr=lr_schedule,
	momentum=beta1,
	variance_momentum=beta2,
	epsilon=1.5e-8,
	gradient_clipping_threshold_per_sample=3.0)
progress_printer = C.logging.ProgressPrinter(tag='Training')
trainer = C.Trainer(model, (overall_loss, rmse_eval), learner, progress_printer)


C.logging.log_number_of_parameters(model) ; print()

data = {input_var: input_imgs_reshaped};

# Get minibatches of images to train with and perform model training
for epoch in range(55):
	sample_count = 0
	while sample_count < epoch_size:  # loop over minibatches in the epoch
		trainer.train_minibatch({input_var: input_imgs_reshaped[np.random.choice(len(input_imgs_reshaped), size=minibatch_size, replace=False)]})
		sample_count += minibatch_size

	trainer.summarize_training_progress()
	test_image(10)

save_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'largedata', "autoencoder_checkpoint")
model.save(save_filename)


test_image(10)


