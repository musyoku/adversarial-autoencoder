import numpy as np
import random
from math import *
from chainer import Variable

def onehot_categorical(batchsize, num_labels):
	y = np.zeros((batchsize, num_labels), dtype=np.float32)
	indices = np.random.randint(0, num_labels, batchsize)
	for b in range(batchsize):
		y[b, indices[b]] = 1
	return y

def uniform(batchsize, ndim, minv=-1, maxv=1):
	return np.random.uniform(minv, maxv, (batchsize, ndim)).astype(np.float32)

def gaussian(batchsize, ndim, mean=0, var=1):
	return np.random.normal(mean, var, (batchsize, ndim)).astype(np.float32)

def gaussian_mixture(batchsize, ndim, num_labels):
	if ndim % 2 != 0:
		raise Exception("ndim must be a multiple of 2.")

	def sample(x, y, label, num_labels):
		shift = 1.4
		r = 2.0 * np.pi / float(num_labels) * float(label)
		new_x = x * cos(r) - y * sin(r)
		new_y = x * sin(r) + y * cos(r)
		new_x += shift * cos(r)
		new_y += shift * sin(r)
		return np.array([new_x, new_y]).reshape((2,))

	x_var = 0.5
	y_var = 0.05
	x = np.random.normal(0, x_var, (batchsize, ndim // 2))
	y = np.random.normal(0, y_var, (batchsize, ndim // 2))
	z = np.empty((batchsize, ndim), dtype=np.float32)
	for batch in range(batchsize):
		for zi in range(ndim // 2):
			z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], random.randint(0, num_labels - 1), num_labels)
	return z

def supervised_gaussian_mixture(batchsize, ndim, label_indices, num_labels):
	if ndim % 2 != 0:
		raise Exception("ndim must be a multiple of 2.")

	def sample(x, y, label, num_labels):
		shift = 1.4
		r = 2.0 * np.pi / float(num_labels) * float(label)
		new_x = x * cos(r) - y * sin(r)
		new_y = x * sin(r) + y * cos(r)
		new_x += shift * cos(r)
		new_y += shift * sin(r)
		return np.array([new_x, new_y]).reshape((2,))

	x_var = 0.5
	y_var = 0.05
	x = np.random.normal(0, x_var, (batchsize, ndim // 2))
	y = np.random.normal(0, y_var, (batchsize, ndim // 2))
	z = np.empty((batchsize, ndim), dtype=np.float32)
	for batch in range(batchsize):
		for zi in range(ndim // 2):
			z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], num_labels)
	return z

def swiss_roll(batchsize, ndim, num_labels):
	def sample(label, num_labels):
		uni = np.random.uniform(0.0, 1.0) / float(num_labels) + float(label) / float(num_labels)
		r = sqrt(uni) * 3.0
		rad = np.pi * 4.0 * sqrt(uni)
		x = r * cos(rad)
		y = r * sin(rad)
		return np.array([x, y]).reshape((2,))

	z = np.zeros((batchsize, ndim), dtype=np.float32)
	for batch in range(batchsize):
		for zi in range(ndim // 2):
			z[batch, zi*2:zi*2+2] = sample(random.randint(0, num_labels - 1), num_labels)
	return z

def supervised_swiss_roll(batchsize, ndim, label_indices, num_labels):
	def sample(label, num_labels):
		uni = np.random.uniform(0.0, 1.0) / float(num_labels) + float(label) / float(num_labels)
		r = sqrt(uni) * 3.0
		rad = np.pi * 4.0 * sqrt(uni)
		x = r * cos(rad)
		y = r * sin(rad)
		return np.array([x, y]).reshape((2,))

	z = np.zeros((batchsize, ndim), dtype=np.float32)
	for batch in range(batchsize):
		for zi in range(ndim // 2):
			z[batch, zi*2:zi*2+2] = sample(label_indices[batch], num_labels)
	return z