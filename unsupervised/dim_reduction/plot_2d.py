import numpy as np
import os, pylab
from model import aae
from args import args
import dataset
import plot

try:
	os.mkdir(args.plot_dir)
except:
	pass

def main():
	images, labels = dataset.load_test_images()
	num_scatter = len(images)

	y_distribution, z = aae.encode_x_yz(images, apply_softmax=False, test=True)
	y = aae.argmax_onehot_from_unnormalized_distribution(y_distribution)
	representation = aae.to_numpy(aae.encode_yz_representation(y, z, test=True))

	plot.scatter_labeled_z(representation, labels, dir=args.plot_dir)
	
if __name__ == "__main__":
	main()
