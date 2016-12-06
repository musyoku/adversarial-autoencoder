# -*- coding: utf-8 -*-
import os, sys, time, pylab
import numpy as np
import matplotlib.patches as mpatches
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
import dataset
from args import args
from model import aae

try:
	os.mkdir(args.plot_dir)
except:
	pass

images, labels = dataset.load_test_images()
config = aae.config
num_clusters = config.ndim_y
num_plots_per_cluster = 11
image_width = 28
image_height = 28
pylab.gray()

# plot cluster head
head_y = np.identity(config.ndim_y, dtype=np.float32)
zero_z = np.zeros((config.ndim_y, config.ndim_z), dtype=np.float32)
head_x = aae.to_numpy(aae.decode_yz_x(head_y, zero_z, test=True))
for n in xrange(num_clusters):
	pylab.subplot(num_clusters, num_plots_per_cluster + 2, n * (num_plots_per_cluster + 2) + 1)
	pylab.imshow(head_x[n].reshape((image_width, image_height)), interpolation="none")
	pylab.axis("off")

# plot elements in cluster
counts = [0 for i in xrange(num_clusters)]
indices = np.arange(len(images))
np.random.shuffle(indices)
batchsize = 500

i = 0
x_batch = np.zeros((batchsize, config.ndim_x), dtype=np.float32)
for n in xrange(len(images) / batchsize):
	for b in xrange(batchsize):
		x_batch[b] = images[indices[i]].reshape((config.ndim_x,))
		i += 1
	labels = aae.argmax_x_label(x_batch, test=True)
	for m in xrange(labels.size):
		cluster = int(labels[m])
		counts[cluster] += 1
		if counts[cluster] <= num_plots_per_cluster:
			x = x_batch[m]
			pylab.subplot(num_clusters, num_plots_per_cluster + 2, cluster * (num_plots_per_cluster + 2) + 2 + counts[cluster])
			pylab.imshow(x.reshape((image_width, image_height)), interpolation="none")
			pylab.axis("off")

fig = pylab.gcf()
fig.set_size_inches(num_plots_per_cluster, num_clusters)
pylab.savefig("{}/clusters.png".format(args.plot_dir))

