# -*- coding: utf-8 -*-
import os, sys, time, pylab
import numpy as np
from chainer import cuda, Variable
import matplotlib.patches as mpatches
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
import util
import visualizer
from args import args
from model import conf, aae

try:
	os.mkdir(args.vis_dir)
except:
	pass

dataset = util.load_images(args.test_image_dir)

n_clusters = conf.ndim_y
n_elements_per_cluster = 11
n_image_channels = 1
image_width = 28
image_height = 28

if n_image_channels == 1:
	pylab.gray()
xp = np
if conf.gpu_enabled:
	xp = cuda.cupy

# visualize reconstruction
x = util.sample_x_variable(100, conf.ndim_x, dataset, gpu_enabled=conf.gpu_enabled)
y_distribution, z = aae.generator_x_yz(x, test=True, apply_f=True)
_x = aae.decoder_yz_x(y_distribution, z, test=True, apply_f=True)
if conf.gpu_enabled:
	z.to_cpu()
	_x.to_cpu()
visualizer.tile_x(_x.data, dir=args.vis_dir)

# plot cluster head
head_y = Variable(xp.identity(conf.ndim_y, dtype=xp.float32))
base_z = Variable(xp.zeros((conf.ndim_y, conf.ndim_z), dtype=xp.float32))
head_x = aae.decoder_yz_x(head_y, base_z, test=True, apply_f=True)
if conf.gpu_enabled:
	head_x.to_cpu()
for n in xrange(n_clusters):
	pylab.subplot(n_clusters, n_elements_per_cluster + 2, n * (n_elements_per_cluster + 2) + 1)
	if n_image_channels == 1:
		pylab.imshow(head_x.data[n].reshape((image_width, image_height)), interpolation="none")
	elif n_image_channels == 3:
		pylab.imshow(head_x.data[n].reshape((n_image_channels, image_width, image_height)), interpolation="none")
	pylab.axis("off")

# plot elements in cluster
counts = [0 for i in xrange(n_clusters)]
indices = np.arange(len(dataset))
np.random.shuffle(indices)
batchsize = 500

i = 0
x_batch_data = np.zeros((batchsize, conf.ndim_x), dtype=np.float32)
for n in xrange(len(dataset) / batchsize):
	for b in xrange(batchsize):
		x_batch_data[b] = dataset[indices[i]].reshape((conf.ndim_x,))
		i += 1
	x_batch = Variable(x_batch_data)	
	if conf.gpu_enabled:
		x_batch.to_gpu()
	y_batch, _ = aae.encode_x_yz(x_batch, test=True, apply_f=True)
	c_batch = xp.argmax(y_batch.data, axis=1)
	print c_batch

	if conf.gpu_enabled:
		x_batch.to_cpu()
	for c in xrange(c_batch.size):
		cluster = int(c_batch[c])
		counts[cluster] += 1
		if counts[cluster] <= n_elements_per_cluster:
			element_x = x_batch[c].data
			pylab.subplot(n_clusters, n_elements_per_cluster + 2, cluster * (n_elements_per_cluster + 2) + 2 + counts[cluster])
			if n_image_channels == 1:
				pylab.imshow(element_x.reshape((image_width, image_height)), interpolation="none")
			elif n_image_channels == 3:
				pylab.imshow(element_x.reshape((n_image_channels, image_width, image_height)), interpolation="none")
			pylab.axis("off")

fig = pylab.gcf()
fig.set_size_inches(n_elements_per_cluster, n_clusters)
pylab.savefig("{}/clusters.png".format(args.vis_dir))

