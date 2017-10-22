
# coding: utf-8

# In[1]:


from IPython.display import display, Image


# In[2]:


from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
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

get_ipython().magic('matplotlib inline')


# In[3]:


C.device.try_set_default_device(C.device.gpu(1))
# Select the right target device when this notebook is being tested:
if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        C.device.try_set_default_device(C.device.cpu())
    else:
        C.device.try_set_default_device(C.device.gpu(0))


# In[4]:


# Ensure we always get the same amount of randomness
np.random.seed(0)


# In[5]:


image_height = 150
image_width  = 150
num_channels = 3
image_dim = image_height * image_width * num_channels
image_shape = (num_channels, image_height, image_width)

x = C.input_variable(image_dim)
y = C.input_variable(image_dim)


# In[142]:


encoded_size = 384

# Input variable and normalization
input_var = C.ops.input_variable(image_shape, np.float32)
scaled_input = C.ops.element_divide(input_var, C.ops.constant(256.), name="input_node")
noisy_scaled_input = C.ops.plus(scaled_input, C.random.normal(image_shape, scale=0.1))

cMap = 16
conv1   = C.layers.Convolution2D  ((3,3), 8, pad=True, activation=C.ops.tanh)(noisy_scaled_input)


conv2   = C.layers.Convolution2D  ((3,3), cMap, pad=True, activation=C.ops.tanh)(C.layers.Dropout(0.05)(conv1))
pconv2   = C.layers.Convolution2D  ((3,3), cMap, strides=(3,3), pad=True, activation=C.ops.tanh)(conv2)
#pool1   = C.layers.MaxPooling   ((3,3), (3,3), name="pooling_node")(conv2)
conv3   = C.layers.Convolution2D  ((5,5), cMap, strides=(2,2), pad=True, activation=C.ops.tanh)(C.layers.Dropout(0.1)(pconv2))
#bn1     = C.layers.BatchNormalization()(conv3)
conv4   = C.layers.Convolution2D  ((5,5), cMap, pad=True, strides=(2,2), activation=C.ops.tanh)(C.layers.Dropout(0.1)(conv3))

fc1     = C.layers.Dense(1024, activation=None)(conv4)
act1    = C.ops.param_relu(C.ops.parameter(shape=1024), fc1)
fc2    = C.layers.Dense(encoded_size, activation=None)(act1)
act2    = C.ops.tanh(fc2)#C.ops.param_relu(C.ops.parameter(shape=encoded_size), fc12)

fc3     = C.layers.Dense(512, activation=C.ops.leaky_relu)(C.ops.placeholder(shape=encoded_size))
fc4     = C.layers.Dense(1536, activation=C.ops.leaky_relu)(fc3)
fc5     = C.layers.Dense(16*50*50, activation=C.ops.leaky_relu)(fc4)

image_shape_onelayer = list(image_shape)
image_shape_onelayer[0] = 1
image_shape_onelayer = tuple(image_shape_onelayer)

rs1     = C.ops.reshape(fc5, (16,50,50))
pdeconv1      = C.layers.ConvolutionTranspose2D((3,3), cMap, strides=(3,3), pad=False, bias=False, init=C.glorot_uniform(1), name="sln")(rs1)
deconv1       = C.layers.ConvolutionTranspose2D((5,5), cMap, pad=True, name="sln")(pdeconv1)

#unpool1 = C.layers.MaxUnpooling ((3,3), (3,3))(pool1, deconv1)
deconv2       = C.layers.ConvolutionTranspose2D((5,5), num_channels, activation=None, pad=True, bias=False, name="output_node")(deconv1)

latent_log_sigma = C.layers.Dense(encoded_size, activation=None)(act2)
latent_mean = C.layers.Dense(encoded_size, activation=C.ops.tanh)(act2)
latent_sigma = C.ops.exp(latent_log_sigma)
latent_vec = C.ops.plus(latent_mean, C.ops.element_times(latent_sigma, C.random.normal_like(latent_log_sigma)))
latent_kl_loss = -0.5 * C.ops.reduce_mean(1 + latent_log_sigma - C.ops.square(latent_mean) - latent_sigma, axis=-1)


z = deconv2(latent_vec)

decode_input = C.ops.input_variable(encoded_size)
decode_output = deconv2(decode_input)

#f2        = C.ops.element_times(C.ops.constant(0.00390625), input_var)
err       = C.ops.reshape(C.ops.minus(z, scaled_input), (image_dim))
sq_err    = C.ops.square(err)
mse       = C.ops.reduce_mean(sq_err)
rmse_loss = C.ops.sqrt(mse)
rmse_eval = rmse_loss

overall_loss = rmse_loss + latent_kl_loss


# In[143]:


#input_img_filenames = ['../data/lfw/all/John_Bolton_00{:02d}.jpg'.format(k) for k in range(1,16)] + ['../data/lfw/all/John_Ashcroft_00{:02d}.jpg'.format(k) for k in range(1,46)] + ['../data/lfw/all/Yoriko_Kawaguchi_00{:02d}.jpg'.format(k) for k in range(1,10)] + ['../data/lfw/all/Joe_Lieberman_00{:02d}.jpg'.format(k) for k in range(1,11)]
input_img_filenames = ['../data/lfw/all/{}'.format(f) for f in os.listdir('../data/lfw/all') if re.match('.*\\.(png|jpg)',f)]
#input_img_filenames = ['/dev/shm/img_128/{}'.format(f) for f in os.listdir('/dev/shm/img_128') if re.match('.*\\.png',f)]
input_imgs_raw = [scipy.ndimage.io.imread(file)[50:-50,50:-50] for file in input_img_filenames]
input_imgs_reshaped = np.array([np.ascontiguousarray(np.rollaxis(np.rollaxis(np.array(img, dtype=np.float32), 0, 3), 0, 3)) for img in input_imgs_raw])
#input_img_batches = np.array_split(input_imgs_reshaped, 8)


# In[144]:


img_num = 5429
#img = np.array(np.ascontiguousarray(np.rollaxis(np.rollaxis(np.array(scipy.ndimage.io.imread('/dev/shm/img_128/img_hdr_1.png')[50:-50,50:-50], dtype=np.float32), 0, 3), 0, 3)))
img = np.clip(input_imgs_reshaped[img_num]/256 + np.random.normal(scale=0.1, size=image_shape), 0.0, 255.0).astype(np.float32)
tmp2 = C.ops.clip(z, 0.0, 1.0).eval({input_var: img})
print(tmp2.shape)
#tmp2 = z.eval({fc1.arguments[0]: np.array(np.random.random(size=15), dtype=np.float32)})
print(rmse_loss.eval({input_var: img}))
output_img = np.rollaxis(np.rollaxis(tmp2.reshape(image_shape), 2, -3), 2, -3)
#output_img = tmp2.reshape((150,150))
plt.rcParams["figure.figsize"] = [12,9]
plt.subplot(1, 3, 1)
#plt.imshow(scipy.ndimage.io.imread('/dev/shm/img_128/img_hdr_1.png')[50:-50,50:-50])
plt.imshow(np.rollaxis(np.rollaxis(img, 2, -3), 2, -3).astype(np.float64))
plt.subplot(1, 3, 2)
plt.imshow(output_img)
plt.subplot(1, 3, 3)
#new_input = np.array([[0.0, 0.0, 0.0, 4.654534339904785, 0.0, 0.0, 0.3437899351119995, 0.0, 0.0, 0.0, 0.0, 1.6117901802062988, 0.0, 0.0, 50.309154033660889, 10.763576507568359, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.140332221984863, 0.0, 20.741523265838623, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 10.149624824523926, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.301928520202637, 0.0, 0.0, 0.0]], dtype=np.float32)
#new_input = np.ones(50) * 4
#new_input[23] = 2
#new_input[25] = 2
#new_input[27] = 2
new_input = (np.array(np.random.normal(size=encoded_size), dtype=np.float32)+0.01)**(2)
#new_input[np.random.randint(encoded_size, size=encoded_size//4)] = 0.5
#tl = np.zeros((1, encoded_size))
#for i in range(3):
#    tl = np.maximum(tl,(act2.eval({input_var: input_imgs_reshaped[np.random.randint(40)]}))*np.random.random(1))
#new_input = tl/1.0
#new_input = np.array([[-0.9512496590614319, -0.856913149356842, 0.9999237656593323, -0.8241662979125977, 0.9942786693572998, -0.9974982738494873, 0.42951700091362, -0.9297400116920471, -0.9332999587059021, -0.9218769669532776, 0.040611058473587036, -0.7809401750564575, -0.9883394241333008, -0.9349163770675659, -0.8004962205886841, 0.8070048093795776, -0.9985208511352539, 0.49706748127937317, -0.6436474919319153, -0.7356144785881042, -0.5454084873199463, -0.7967053651809692, 0.4403694272041321, -0.9526838064193726, -0.9792911410331726, 0.9842500686645508, -0.9331961870193481, -0.9552119970321655, -0.9333693981170654, 0.9273069500923157, -0.8993655443191528, -0.9599452018737793, 0.8836828470230103, 0.9200283885002136, 0.2272624373435974, -0.9917638897895813, 0.9793252348899841, -0.7675734758377075, 0.9956253170967102, 0.9947523474693298, -0.9692676067352295, 0.9881631135940552, 0.1569795310497284, 0.9375765919685364, 0.9918071627616882, 0.8698028326034546, -0.7092040777206421, 0.9984179735183716]], dtype=np.float32)
#new_input += (np.array(np.random.random(size=encoded_size), dtype=np.float32))*1
tmp3 = C.ops.clip(decode_output, 0.0, 1.0).eval({decode_input: new_input})[0][:3]
output_img2 = np.rollaxis(np.rollaxis(tmp3.reshape(image_shape), 2, -3), 2, -3)
plt.imshow(output_img2)

def encode(img_data):
    return latent_mean.eval({input_var: img_data})

#print(encode(img).tolist())


# In[ ]:


# training config
max_epochs = 12
epoch_size = 2048
minibatch_size = 48

mn = 0

# Set learning parameters
lr_schedule = C.learning_rate_schedule([0.01], C.learners.UnitType.sample, epoch_size)
mm_schedule = C.learners.momentum_as_time_constant_schedule([1900], epoch_size)

# Instantiate the trainer object to drive the model training
#learner = C.learners.nesterov(z.parameters, lr_schedule, mm_schedule, unit_gain=True)
#learner = C.learners.adadelta(z.parameters)
learning_rate = 1.5e-3
lr_schedule = C.learning_rate_schedule([learning_rate * (0.97**i) for i in range(30)], C.UnitType.sample, epoch_size=12000)
beta1 = C.momentum_schedule(0.88)
beta2 = C.momentum_schedule(0.9991)
learner = C.adam(z.parameters,
                 lr=lr_schedule,
                 momentum=beta1,
                 variance_momentum=beta2,
                 epsilon=1.5e-8,
                 gradient_clipping_threshold_per_sample=3.0)
progress_printer = C.logging.ProgressPrinter(tag='Training')
trainer = C.Trainer(z, (overall_loss, rmse_eval), learner, progress_printer)



# In[ ]:


C.logging.log_number_of_parameters(z) ; print()

data = {input_var: input_imgs_reshaped};

# Get minibatches of images to train with and perform model training
for epoch in range(20):       # loop over epochs
    sample_count = 0
    while sample_count < epoch_size:  # loop over minibatches in the epoch
        #print(data)
        trainer.train_minibatch({input_var: input_imgs_reshaped[np.random.randint(len(input_imgs_reshaped), size=minibatch_size)]})                                   # update model with it
        sample_count += minibatch_size #data[input_var].num_samples                     # count samples processed so far


    trainer.summarize_training_progress()


# In[ ]:


img_lookup_table = {}
for i in range(len(input_imgs_reshaped)):
    img_lookup_table[i] = encode(input_imgs_reshaped[i]).reshape(-1)
    if i % 1000 == 0:
        print('Added a total of {} images to the table'.format(i))

def lookup_img(img_data, k_max=1):
    encoded_image = encode(img_data).reshape(-1)
    min_dist_keys = []
    i = 0
    for key in img_lookup_table:
        diff = encoded_image - img_lookup_table[key]
        dist = np.dot(diff, diff)
        min_dist_keys.append((dist, key))
        if i % (k_max + 500) == 0:
            min_dist_keys = sorted(min_dist_keys)[:k_max]
    return sorted(min_dist_keys)[:k_max]


# In[ ]:


lookup_img(input_imgs_reshaped[4], k_max=6)


# In[13]:


def lookup_img(img_data, k_max=1):
    encoded_image = encode(img_data).reshape(-1)
    min_dist_keys = []
    i = 0
    for key in img_lookup_table:
        diff = encoded_image - img_lookup_table[key]
        dist = np.dot(diff, diff)
        min_dist_keys.append((dist, key))
        if i % (k_max + 500) == 0:
            min_dist_keys = sorted(min_dist_keys)[:k_max]
    return sorted(min_dist_keys)[:k_max]


# In[93]:


aa=np.rollaxis(np.rollaxis(img, 0, 3), 2, 3)

#print(np.max(img), np.min(img))
print(np.rollaxis(np.rollaxis(img, 2, -3), 2, -3)[:10][:10])


# In[ ]:





# In[ ]:




