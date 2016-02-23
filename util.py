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
	z = np.random.uniform(-1, 1, (batchsize, z_dimension)).astype(np.float32)
	if gpu:
		z = cuda.to_gpu(z)
	z = Variable(z)
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
	
	if gpu:
		z = cuda.to_gpu(z)
	return Variable(z)

def random_sampling(args, img_channel, img_width, n_z, dataset, generator, decoder, use_gpu=True):
	pylab.rcParams["figure.figsize"] = (16.0,16.0)
	pylab.clf()
	if img_channel == 1:
		pylab.gray()
	z = sample_z_from_noise_prior(100, n_z, use_gpu);
	x = decoder(z, test=True)
	if use_gpu:
		x.to_cpu()
	for m in range(100):
		pylab.subplot(10, 10, m + 1)
		if img_channel == 1:
			pylab.imshow(np.clip((x.data[m] + 1.0) / 2.0, 0.0, 1.0).reshape((img_width, img_width)), interpolation="none")
		elif img_channel == 3:
			pylab.imshow(np.clip((x.data[m] + 1.0) / 2.0, 0.0, 1.0).transpose(1, 2, 0), interpolation="none")
		pylab.axis("off")
	pylab.savefig("%s/random.png" % args.visualization_dir)

def transform_linear(args, img_channel, img_width, n_z, dataset, generator, decoder, use_gpu=True):
	x_batch = np.zeros((1, img_channel, img_width, img_width), dtype=np.float32)
	data_index = np.random.randint(len(dataset))
	img = dataset[data_index]
	if img_channel == 1:
		x_batch[0,0,:,:] = img[:,:]
	elif img_channel == 3:
		x_batch[0,:,:,:] = img[:,:,:]

	x_batch = Variable(x_batch)
	if use_gpu:
		x_batch.to_gpu()
	
	z_batch = generator(x_batch, test=True)
	if use_gpu:
		z_batch.to_cpu()

	pylab.rcParams["figure.figsize"] = (16.0,16.0)
	pylab.clf()
	if img_channel == 1:
		pylab.gray()
	
	pylab.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.1)

	max_dimention = 2

	for col in xrange(21):
		new_z_batch = np.zeros((max_dimention, n_z), dtype=np.float32)
		for n in xrange(max_dimention):
			new_z_batch[n,:] = z_batch.data[0]
			new_z = new_z_batch[n]
			elem = -1.0 * col / 20.0 + 1.0 * (1.0 - col / 20.0) + new_z[n]
			new_z[n] = elem
		new_z_batch = Variable(new_z_batch)
		if use_gpu:
			new_z_batch.to_gpu()
		new_x = decoder(new_z_batch, test=True)
		if use_gpu:
			new_x.to_cpu()
		for n in xrange(max_dimention):
			pylab.subplot(max_dimention, 21, n * 21 + col + 1)
			if img_channel == 1:
				pylab.imshow(np.clip((new_x.data[n] + 1.0) / 2.0, 0.0, 1.0).reshape((img_width, img_width)), interpolation="none")
			elif img_channel == 3:
				pylab.imshow(np.clip((new_x.data[n] + 1.0) / 2.0, 0.0, 1.0).transpose(1, 2, 0), interpolation="none")
			pylab.axis("off")
				
	pylab.savefig("%s/transform.png" % args.visualization_dir)




def visualize_pz(args, z_batch, label_batch):
	pylab.rcParams["figure.figsize"] = (7.0, 7.0)
	pylab.clf()
	colors = ["#e40402","#05aaa8","#ac02ab","#aba808","#151515","#94a169", "#bec9cd", "#0e960e", "#6a6551","#2103c8"]
	for n in xrange(z_batch.shape[0]):
		pylab.scatter(z_batch[n, 0], z_batch[n, 1], c=colors[label_batch[n]], s=20, marker="o", edgecolors='none')
	

	pylab.xticks(pylab.arange(-2, 3))
	pylab.yticks(pylab.arange(-2, 3))
	pylab.xlabel("z1")
	pylab.ylabel("z2")
	pylab.savefig("%s/pz.png" % args.visualization_dir)

# def visualize_pz(z_batch, label_batch):
# 	pylab.rcParams["figure.figsize"] = (7.0, 7.0)
# 	colors = ["#e40402","#05aaa8","#ac02ab","#aba808","#151515","#94a169", "#bec9cd", "#0e960e", "#6a6551","#2103c8"]
# 	for label in xrange(10):
# 		label_indices = np.ones((100,)) * label
# 		z = sample_z_from_10_2d_gaussian_mixture(100, label_indices)
# 		z = z.data
# 		for i in xrange(100):
# 			pylab.scatter(z[i, 0], z[i, 1], c=colors[label], s=20, marker="o", edgecolors='none')
	

# 	pylab.xticks(pylab.arange(-2, 3))
# 	pylab.yticks(pylab.arange(-2, 3))
# 	pylab.xlabel("z1")
# 	pylab.ylabel("z2")
# 	pylab.show()