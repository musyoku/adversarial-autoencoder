from __future__ import division
from __future__ import print_function
import os, pylab, chainer, argparse
import numpy as np
from model import Model
from aae.utils import onehot
import aae.plot as plot
import aae.sampler as sampler

def plot_clusters():
	dataset_train, dataset_test = chainer.datasets.get_mnist()
	images_train, labels_train = dataset_train._datasets
	images_test, labels_test = dataset_test._datasets
	dataset_indices = np.arange(0, len(images_test))
	np.random.shuffle(dataset_indices)

	model = Model()
	assert model.load("model.hdf5")

	# normalize
	images_train = (images_train - 0.5) * 2
	images_test = (images_test - 0.5) * 2

	num_clusters = model.ndim_y
	num_plots_per_cluster = 11
	image_width = 28
	image_height = 28
	ndim_x = image_width * image_height
	pylab.gray()

	with chainer.no_backprop_mode() and chainer.using_config("train", False):
		# plot cluster head
		head_y = np.identity(model.ndim_y, dtype=np.float32)
		zero_z = np.zeros((model.ndim_y, model.ndim_z), dtype=np.float32)
		head_x = model.decode_yz_x(head_y, zero_z).data
		head_x = (head_x + 1.0) / 2.0
		for n in range(num_clusters):
			pylab.subplot(num_clusters, num_plots_per_cluster + 2, n * (num_plots_per_cluster + 2) + 1)
			pylab.imshow(head_x[n].reshape((image_width, image_height)), interpolation="none")
			pylab.axis("off")

		# plot elements in cluster
		counts = [0 for i in range(num_clusters)]
		indices = np.arange(len(images_test))
		np.random.shuffle(indices)
		batchsize = 500

		i = 0
		x_batch = np.zeros((batchsize, ndim_x), dtype=np.float32)
		for n in range(len(images_test) // batchsize):
			for b in range(batchsize):
				x_batch[b] = images_test[indices[i]]
				i += 1
			y_batch = model.encode_x_yz(x_batch)[0].data
			labels = np.argmax(y_batch, axis=1)
			for m in range(labels.size):
				cluster = int(labels[m])
				counts[cluster] += 1
				if counts[cluster] <= num_plots_per_cluster:
					x = (x_batch[m] + 1.0) / 2.0
					pylab.subplot(num_clusters, num_plots_per_cluster + 2, cluster * (num_plots_per_cluster + 2) + 2 + counts[cluster])
					pylab.imshow(x.reshape((image_width, image_height)), interpolation="none")
					pylab.axis("off")

		fig = pylab.gcf()
		fig.set_size_inches(num_plots_per_cluster, num_clusters)
		pylab.savefig("clusters.png")

def plot_scatter():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", "-m", type=str, default="model.hdf5")
	args = parser.parse_args()

	dataset_train, dataset_test = chainer.datasets.get_mnist()
	images_train, labels_train = dataset_train._datasets
	images_test, labels_test = dataset_test._datasets

	model = Model()
	assert model.load(args.model)

	# normalize
	images_train = (images_train - 0.5) * 2
	images_test = (images_test - 0.5) * 2

	with chainer.no_backprop_mode() and chainer.using_config("train", False):
		z = model.encode_x_yz(images_test)[1].data
	plot.scatter_labeled_z(z, labels_test, "scatter_gen.png")
	
	
if __name__ == "__main__":
	plot_clusters()
	plot_scatter()
