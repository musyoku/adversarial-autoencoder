import numpy as np
import os, pylab
from model import aae
from args import args
import dataset
import visualizer

try:
	os.mkdir(args.plot_dir)
except:
	pass

def main():
	# load MNIST images
	images, labels = dataset.load_test_images()

	# config
	config = aae.config
	num_scatter = len(images)

	x, _, labels = dataset.sample_labeled_data(images, labels, num_scatter, config.ndim_x, config.ndim_y)
	y_distribution, z = aae.encode_x_yz(x, apply_softmax=False, test=True)
	y = aae.argmax_onehot_from_unnormalized_distribution(y_distribution)
	representation = aae.to_numpy(aae.encode_yz_representation(y, z, test=True))

	visualizer.plot_labeled_z(representation, labels, dir=args.plot_dir)
	
if __name__ == "__main__":
	main()
