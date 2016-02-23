# -*- coding: utf-8 -*-
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
import os, sys, time
sys.path.append(os.path.split(os.getcwd())[0])
from args import args
from config import config
from model import gen, dis, dec
from util import *

def sample_x_from_data_distribution(batchsize):
	shape = config.img_channel * config.img_width * config.img_width
	x_batch = np.zeros((batchsize, shape), dtype=np.float32)
	for j in range(batchsize):
		data_index = np.random.randint(len(dataset))
		img = dataset[data_index]
		x_batch[j] = img.reshape((shape,))
	x_batch = Variable(x_batch)
	if use_gpu:
		x_batch.to_gpu()
	return x_batch

def sample_x_and_label_from_data_distribution(batchsize):
	shape = config.img_channel * config.img_width * config.img_width
	x_batch = np.zeros((batchsize, shape), dtype=np.float32)
	label_batch = np.zeros((batchsize, 1), dtype=np.int32)
	for j in range(batchsize):
		data_index = np.random.randint(len(dataset))
		img = dataset[data_index]
		x_batch[j] = img.reshape((shape,))
		label_batch[j] = labels[data_index]
	x_batch = Variable(x_batch)
	label_batch = Variable(label_batch)
	if config.use_gpu:
		x_batch.to_gpu()
		label_batch.to_gpu()
	return x_batch, label_batch

def visualize_reconstruction():
	x_batch = sample_x_from_data_distribution(100)
	
	z_batch = gen(x_batch, test=True)
	_x_batch = dec(z_batch, test=True)
	if use_gpu:
		_x_batch.to_cpu()

	pylab.rcParams["figure.figsize"] = (16.0,16.0)
	pylab.clf()
	if config.img_channel == 1:
		pylab.gray()
	for m in range(100):
		pylab.subplot(10, 10, m + 1)
		if config.img_channel == 1:
			pylab.imshow(np.clip((_x_batch.data[m] + 1.0) / 2.0, 0.0, 1.0).reshape((config.img_width, config.img_width)), interpolation="none")
		elif config.img_channel == 3:
			pylab.imshow(np.clip((_x_batch.data[m] + 1.0) / 2.0, 0.0, 1.0).reshape((config.img_channel, config.img_width, config.img_width)), interpolation="none")
		pylab.axis("off")
	pylab.savefig("%s/reconstruction.png" % args.visualization_dir)

def visualize_walkthrough():
	x_batch = sample_x_from_data_distribution(20)
	z_batch = gen(x_batch, test=True)
	if use_gpu:
		z_batch.to_cpu()

	pylab.rcParams["figure.figsize"] = (16.0,16.0)
	pylab.clf()
	if config.img_channel == 1:
		pylab.gray()
	
	z_a = z_batch.data[:10,:]
	z_b = z_batch.data[10:,:]
	for col in range(10):
		_z_batch = z_a * (1 - col / 9.0) + z_b * col / 9.0
		_z_batch = Variable(_z_batch)
		if use_gpu:
			_z_batch.to_gpu()
		_x_batch = dec(_z_batch, test=True)
		if use_gpu:
			_x_batch.to_cpu()
		for row in range(10):
			pylab.subplot(10, 10, row * 10 + col + 1)
			if config.img_channel == 1:
				pylab.imshow(np.clip((_x_batch.data[row] + 1.0) / 2.0, 0.0, 1.0).reshape((config.img_width, config.img_width)), interpolation="none")
			elif config.img_channel == 3:
				pylab.imshow(np.clip((_x_batch.data[row] + 1.0) / 2.0, 0.0, 1.0).reshape((config.img_channel, config.img_width, config.img_width)), interpolation="none")
			pylab.axis("off")
				
	pylab.savefig("%s/walk_through.png" % args.visualization_dir)

def visualize_labeled_z():
	x_batch, label_batch = sample_x_and_label_from_data_distribution(1000)
	z_batch = gen(x_batch, test=True)
	if z_batch.data[0].shape[0] != 2:
		raise Exception("隠れ変数ベクトルは2次元である必要があります")
		
	pylab.rcParams["figure.figsize"] = (7.0, 7.0)
	pylab.clf()
	colors = ["#e40402","#05aaa8","#ac02ab","#aba808","#151515","#94a169", "#bec9cd", "#0e960e", "#6a6551","#2103c8"]
	for n in xrange(z_batch.shape[0]):
		pylab.scatter(z_batch[n, 0], z_batch[n, 1], c=colors[label_batch[n]], s=20, marker="o", edgecolors='none')

	pylab.xticks(pylab.arange(-2, 3))
	pylab.yticks(pylab.arange(-2, 3))
	pylab.xlabel("z1")
	pylab.ylabel("z2")
	pylab.savefig("%s/labeled_z.png" % args.visualization_dir)

try:
	os.mkdir(args.visualization_dir)
except:
	pass

if args.load_epoch == 0:
	raise Exception("load_epochを指定してモデルを読み込む必要があります")

dataset, label = load_mnist_dataset(args)
use_gpu = False

if use_gpu == False:
	gen.to_cpu()
	dec.to_cpu()

visualize_reconstruction()
visualize_walkthrough()