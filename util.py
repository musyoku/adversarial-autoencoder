# -*- coding: utf-8 -*-
import os, re, math, pylab
from math import *
import numpy as np
from StringIO import StringIO
from PIL import Image
from chainer import cuda, Variable, function
from chainer.utils import type_check

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

def load_labeled_dataset(args, convert_to_grayscale=True):
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

def sample_z_from_noise_prior(batchsize, z_dim, gpu=False):
	z = np.random.uniform(-2, 2, (batchsize, z_dim)).astype(np.float32)
	z = Variable(z)
	if gpu:
		z.to_gpu()
	return z

def sample_z_from_n_2d_gaussian_mixture(batchsize, z_dim, label_indices, n_labels, gpu=False):
	if z_dim % 2 != 0:
		raise Exception("z_dim must be a multiple of 2.")

	def sample(x, y, label, n_labels):
		shift = 1.4
		r = 2.0 * np.pi / float(n_labels) * float(label)
		new_x = x * cos(r) - y * sin(r)
		new_y = x * sin(r) + y * cos(r)
		new_x += shift * cos(r)
		new_y += shift * sin(r)
		return np.array([new_x, new_y]).reshape((2,))

	x_var = 0.5
	y_var = 0.05
	x = np.random.normal(0, x_var, (batchsize, z_dim / 2))
	y = np.random.normal(0, y_var, (batchsize, z_dim / 2))
	z = np.empty((batchsize, z_dim), dtype=np.float32)
	for batch in xrange(batchsize):
		for zi in xrange(z_dim / 2):
			z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)

	z = Variable(z)
	if gpu:
		z.to_gpu()
	return z

def sample_z_from_swiss_roll_distribution(batchsize, z_dim, label_indices, n_labels, gpu=False):
	def sample(label, n_labels):
		uni = np.random.uniform(0.0, 1.0) / float(n_labels) + float(label) / float(n_labels)
		r = math.sqrt(uni) * 3.0
		rad = np.pi * 4.0 * math.sqrt(uni)
		x = r * cos(rad)
		y = r * sin(rad)
		return np.array([x, y]).reshape((2,))

	z = np.zeros((batchsize, z_dim), dtype=np.float32)
	for batch in xrange(batchsize):
		for zi in xrange(z_dim / 2):
			z[batch, zi*2:zi*2+2] = sample(label_indices[batch], n_labels)
	
	z = Variable(z)
	if gpu:
		z.to_gpu()
	return z

class Adder(function.Function):
	def check_type_forward(self, in_types):
		n_in = in_types.size()
		type_check.expect(n_in == 2)
		x_type, label_type = in_types

		type_check.expect(
			x_type.dtype == np.float32,
			label_type.dtype == np.float32,
			x_type.ndim == 2,
			label_type.ndim == 2,
		)

	def forward(self, inputs):
		xp = cuda.get_array_module(inputs[0])
		x, label = inputs
		n_batch = x.shape[0]
		output = xp.empty((n_batch, x.shape[1] + label.shape[1]), dtype=xp.float32)
		output[:,:x.shape[1]] = x
		output[:,x.shape[1]:] = label
		return output,

	def backward(self, inputs, grad_outputs):
		x, label = inputs
		return grad_outputs[0][:,:x.shape[1]], grad_outputs[0][:,x.shape[1]:]

def add_label(x, label):
	return Adder()(x, label)