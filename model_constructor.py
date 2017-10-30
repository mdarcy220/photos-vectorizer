#!/usr/bin/python3

import numpy as np
import cntk as C

def construct_model(image_height, image_width, num_channels, encoded_size=512):
	image_dim = image_height * image_width * num_channels
	image_shape = (num_channels, image_height, image_width)

	x = C.input_variable(image_dim)
	y = C.input_variable(image_dim)

	# Input variable and normalization
	input_var = C.ops.input_variable(image_shape, dtype=np.float32, name='image_input')
	scaled_input = input_var  #C.ops.element_divide(input_var, C.ops.constant(256.), name="scaled_input")
	noisy_scaled_input = C.ops.plus(scaled_input, C.random.normal(image_shape, scale=0.02), name='noisy_scaled_input')

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
	fc4 = C.layers.Dense(2048, activation=C.ops.leaky_relu)(fc3)
	fc5 = C.layers.Dense(cMap*42*42, activation=C.ops.leaky_relu)(fc4)

	rs1 = C.ops.reshape(fc5, (cMap,42,42))
	pdeconv1 = C.layers.ConvolutionTranspose2D((5,5), cMap, strides=(3,3), pad=False, bias=True, init=C.glorot_uniform(1), name="sln")(rs1)
	deconv1 = C.layers.ConvolutionTranspose2D((5,5), cMap, strides=(1,1), pad=True, name="sln2")(pdeconv1)

	deconv2 = C.layers.ConvolutionTranspose2D((5,5), num_channels, activation=None, pad=True, bias=False)(deconv1)

	latent_log_sigma = C.layers.Dense(encoded_size, activation=None, name='latent_log_sigma')(act2)
	latent_mean = C.layers.Dense(encoded_size, activation=C.ops.tanh, name='latent_mean')(act2)
	latent_sigma = C.ops.exp(latent_log_sigma, name='latent_sigma')
	latent_vec = C.ops.plus(latent_mean, C.ops.element_times(latent_sigma, C.random.normal_like(latent_log_sigma)))
	latent_kl_loss = -0.5 * C.ops.reduce_mean(1 + latent_log_sigma - C.ops.square(latent_mean) - latent_sigma, axis=-1)
	latent_kl_loss.set_name('latent_kl_loss')


	z = deconv2(latent_vec)

	decode_input = C.ops.input_variable(encoded_size)
	decode_output = deconv2(decode_input)
	decode_output.set_name('decode_output')
	

	err	  = C.ops.minus(z, scaled_input)
	sq_err	  = C.ops.square(err)
	mse	  = C.ops.reduce_mean(sq_err, name='mse')
	rmse_loss = C.ops.sqrt(mse, name='rmse')

	overall_loss = rmse_loss + latent_kl_loss

	return ((z, decode_output), input_var, overall_loss)

