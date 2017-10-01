from __future__ import division
from __future__ import print_function
import os, pylab, chainer, argparse
import numpy as np
from model import Model
from aae.utils import onehot
import aae.plot as plot
import aae.sampler as sampler

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
	plot.scatter_labeled_z(z, labels_test, "scatter_gen.png")
	
	
if __name__ == "__main__":
	plot_scatter()
