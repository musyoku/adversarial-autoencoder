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
		z_batch = model.encode_x_z(x_batch).data

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
			gen_x = model.decode_yz_x(all_y, fixed_z).data
			gen_x = (gen_x + 1.0) / 2.0
			# plot images generated from each label
			for n in range(10):
				pylab.subplot(num_analogies, 10 + 2, m * 12 + 3 + n)
				pylab.imshow(gen_x[n].reshape((28, 28)), interpolation="none")
				pylab.axis("off")

	fig = pylab.gcf()
	fig.set_size_inches(num_analogies, 10)
	pylab.savefig("analogy.png")

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
		z = model.encode_x_z(images_test).data
	plot.scatter_labeled_z(z, labels_test, "scatter_z.png")
	
	
if __name__ == "__main__":
	plot_analogy()
	plot_scatter()
