# -*- coding: utf-8 -*-
import os
import re
from math import *
import pylab
import numpy as np
from StringIO import StringIO
from PIL import Image
from chainer import cuda, Variable

def load_images(args, convert_to_grayscale=True):
	dataset = []
	fs = os.listdir(args.image_dir)
	print "loading", len(fs), "images..."
	for fn in fs:
		f = open("%s/%s" % (args.image_dir, fn), "rb")
		if convert_to_grayscale:
			img = np.asarray(Image.open(StringIO(f.read())).convert("L"), dtype=np.float32) / 255.0
		else:
			img = np.asarray(Image.open(StringIO(f.read())).convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
		img = (img - 0.5) / 0.5
		dataset.append(img)
		f.close()
	return dataset

def load_mnist_dataset(args, convert_to_grayscale=True):
	dataset = []
	labels = []
	fs = os.listdir(args.image_dir)
	print "loading", len(fs), "images..."
	for fn in fs:
		m = re.match("(.)_.+", fn)
		label = int(m.group(1))
		f = open("%s/%s" % (args.image_dir, fn), "rb")
		if convert_to_grayscale:
			img = np.asarray(Image.open(StringIO(f.read())).convert("L"), dtype=np.float32) / 255.0
		else:
			img = np.asarray(Image.open(StringIO(f.read())).convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
		img = (img - 0.5) / 0.5
		dataset.append(img)
		labels.append(label)
		f.close()
	return dataset, labels

def sample_z_from_noise_prior(batchsize, z_dimension, gpu=False):
	z = np.random.uniform(-2, 2, (batchsize, z_dimension)).astype(np.float32)
	z = Variable(z)
	if gpu:
		z.to_gpu()
	return z

def sample_z_from_10_2d_gaussian_mixture(batchsize, label_indices, n_labels, gpu=False):
	def sample(z, label, n_labels):
		shift = 1.4
		r = 2.0 * np.pi / float(n_labels) * float(label)
		x = z[0] * cos(r) - z[1] * sin(r)
		y = z[0] * sin(r) + z[1] * cos(r)
		x += shift * cos(r)
		y += shift * sin(r)
		return np.array([x, y]).reshape((2,))

	x_var = 0.5
	y_var = 0.05
	x = np.random.normal(0, x_var, (batchsize, 1))
	y = np.random.normal(0, y_var, (batchsize, 1))
	z = np.zeros((batchsize, 2), dtype=np.float32)
	for batch in xrange(batchsize):
		z[batch] = sample(np.array([x[batch], y[batch]]), label_indices[batch], n_labels)
	
	z = Variable(z)
	if gpu:
		z.to_gpu()
	return z
