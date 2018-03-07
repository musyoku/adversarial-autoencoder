from __future__ import division
from __future__ import print_function
import os, pylab, chainer, argparse
import numpy as np
from model import Model
from aae.utils import onehot
import aae.plot as plot
import aae.sampler as sampler

def plot_analogy():
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

	num_analogies = 10
	pylab.gray()

	batch_indices = dataset_indices[:num_analogies]
	x_batch = images_test[batch_indices]
	y_batch = labels_test[batch_indices]
	y_onehot_batch = onehot(y_batch)

	with chainer.no_backprop_mode() and chainer.using_config("train", False):
		z_batch = model.encode_x_yz(x_batch)[1].data

		# plot original image on the left
		x_batch = (x_batch + 1.0) / 2.0
		for m in range(num_analogies):
			pylab.subplot(num_analogies, 10 + 2, m * 12 + 1)
			pylab.imshow(x_batch[m].reshape((28, 28)), interpolation="none")
			pylab.axis("off")

		all_y = np.identity(10, dtype=np.float32)
		for m in range(num_analogies):
			# copy z_batch as many as the number of classes
			fixed_z = np.repeat(z_batch[m].reshape(1, -1), 10, axis=0)
			representation = model.encode_yz_representation(all_y, fixed_z)
			gen_x = model.decode_representation_x(representation).data
			gen_x = (gen_x + 1.0) / 2.0
			# plot images generated from each label
			for n in range(10):
				pylab.subplot(num_analogies, 10 + 2, m * 12 + 3 + n)
				pylab.imshow(gen_x[n].reshape((28, 28)), interpolation="none")
				pylab.axis("off")

	fig = pylab.gcf()
	fig.set_size_inches(num_analogies, 10)
	pylab.savefig("analogy.png")

def plot_mapped_representation():
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
		y_onehot, z = model.encode_x_yz(images_test, apply_softmax_y=True)
		representation = model.encode_yz_representation(y_onehot, z)
		mapped_representation = model.linear_transformation(representation)
	plot.scatter_labeled_z(mapped_representation.data, labels_test, "scatter_r.png")
	
def plot_mapped_cluster_head():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", "-m", type=str, default="model.hdf5")
	args = parser.parse_args()

	model = Model()
	assert model.load(args.model)

	identity = np.identity(model.ndim_y, dtype=np.float32)
	mapped_head = model.linear_transformation(identity)

	labels = [i for i in range(10)]
	plot.scatter_labeled_z(mapped_head.data, labels, "cluster_head.png")

if __name__ == "__main__":
	plot_mapped_cluster_head()
	plot_mapped_representation()
	plot_analogy()
