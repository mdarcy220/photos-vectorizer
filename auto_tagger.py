import numpy as np
import cntk as C
import scipy.ndimage
import skimage.transform
import os
import sys

import MySQLdb
conn = MySQLdb.connect('127.0.0.1', 'root', 'DM44DoJ8alquuShI', 'Photos')

def init_labels_map():
	filename = 'imagenet_classes.txt';
	labels_map = []
	with open(filename) as f:
		for line in f.readlines():
			adjusted_line = line[:-1]
			labels_map.append(adjusted_line.split(', '))
	return labels_map

def fix_img_size(img_data):
	return skimage.transform.resize(img_data[:,:,:3], (224,224), mode='reflect')

def load_image(filename):
	img_data = fix_img_size(scipy.ndimage.io.imread(filename))*256.0 - 128.0
	return np.ascontiguousarray(np.rollaxis(np.rollaxis(np.array(img_data, dtype=np.float32), 0, 3), 0, 3))


# For example, labels_map[0] = "tench, Tinca tinca" and so on.
labels_map = init_labels_map()

model_filename = os.path.join(os.path.dirname(__file__), 'saved_model.dnn')
try:
	raw_model = C.load_model(model_filename)
except ValueError:
	print('Failed to load the auto-tagger model. Try running download_resnet.py', file=sys.stderr)
	sys.exit(1)

# Some specific steps for the saved resnet available from the CNTK repo. That
# network includes extra inputs/outputs that we can get rid of to leave just
# the classifier (classifier is named "z" in their saved model)
model = C.combine(raw_model.find_by_name('z'))
model_input_variable  = model.arguments[0]

def autotag_image(image_filename):
	input_image = load_image(image_filename)
	onehot_output = model.eval({model_input_variable: input_image})

	# Suppose onehot_output[6] is the largest element; then best_class=6
	best_class = np.argmax(onehot_output)

	mytags = labels_map[best_class]

	return mytags

# Autotags all new (i.e., not yet tagged) images in the database
def autotag_new_images():
	cur = conn.cursor()
	num_results = cur.execute("SELECT folder1, folder2, sys_file, id FROM Photo photo WHERE NOT EXISTS (SELECT 1 FROM ImageTag tag WHERE tag.image_id = photo.id)", [])
	for i in range(num_results):
		folder1, folder2, sys_file, image_id = cur.fetchone()
		filename = os.path.join("/var/www/uploads", folder1, folder2, sys_file)
		tags = autotag_image(filename)
		cur2 = conn.cursor()
		tmp = cur2.executemany('INSERT INTO ImageTag(image_id, tag) VALUES (%s, %s)', [(image_id, tag) for tag in tags])
		conn.commit()
	print('Auto-tagged {} new images'.format(num_results))

if __name__ == '__main__':
	autotag_new_images()
