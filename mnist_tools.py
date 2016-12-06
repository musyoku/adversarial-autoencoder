# -*- coding: utf-8 -*-
import gzip, os, six, sys
from six.moves.urllib import request
from PIL import Image
import numpy as np

parent = "http://yann.lecun.com/exdb/mnist"
train_images_filename = "train-images-idx3-ubyte.gz"
train_labels_filename = "train-labels-idx1-ubyte.gz"
test_images_filename = "t10k-images-idx3-ubyte.gz"
test_labels_filename = "t10k-labels-idx1-ubyte.gz"
n_train = 60000
n_test = 10000
dim = 28 * 28

def load_mnist(data_filename, label_filename, num):
	images = np.zeros(num * dim, dtype=np.uint8).reshape((num, dim))
	label = np.zeros(num, dtype=np.uint8).reshape((num, ))
	with gzip.open(data_filename, "rb") as f_images, gzip.open(label_filename, "rb") as f_labels:
		f_images.read(16)
		f_labels.read(8)
		for i in six.moves.range(num):
			label[i] = ord(f_labels.read(1))
			for j in six.moves.range(dim):
				images[i, j] = ord(f_images.read(1))

			if i % 100 == 99 or i == num - 1:
				sys.stdout.write("\rloading images ... ({} / {})".format(i + 1, num))
				sys.stdout.flush()
	sys.stdout.write("\n")
	return images, label

def load_train_images():
	if not os.path.exists("../../" + train_images_filename):
		download_mnist_data()
	images, labels = load_mnist("../../" + train_images_filename, "../../" + train_labels_filename, n_train)
	return images, labels

def load_test_images():
	if not os.path.exists("../../" + test_images_filename):
		download_mnist_data()
	images, labels = load_mnist("../../" + test_images_filename, "../../" + test_labels_filename, n_test)
	return images, labels

def download_mnist_data():
	print("Downloading {} ...".format(train_images_filename))
	request.urlretrieve("{}/{}".format(parent, train_images_filename), "../../" + train_images_filename)
	print("Downloading {} ...".format(train_labels_filename))
	request.urlretrieve("{}/{}".format(parent, train_labels_filename), "../../" + train_labels_filename)
	print("Downloading {} ...".format(test_images_filename))
	request.urlretrieve("{}/{}".format(parent, test_images_filename), "../../" + test_images_filename)
	print("Downloading {} ...".format(test_labels_filename))
	request.urlretrieve("{}/{}".format(parent, test_labels_filename), "../../" + test_labels_filename)
	print("Done")

def extract_bitmaps():
	train_dir = "train_images"
	test_dir = "test_images"
	try:
		os.mkdir(train_dir)
		os.mkdir(test_dir)
	except:
		pass
	data_train, label_train = load_test_images()
	data_test, label_test = load_test_images()
	print "Saving training images ..."
	for i in xrange(data_train.shape[0]):
		image = Image.fromarray(data_train[i].reshape(28, 28))
		image.save("{}/{}_{}.bmp".format(train_dir, label_train[i], i))
	print "Saving test images ..."
	for i in xrange(data_test.shape[0]):
		image = Image.fromarray(data_test[i].reshape(28, 28))
		image.save("{}/{}_{}.bmp".format(test_dir, label_test[i], i))