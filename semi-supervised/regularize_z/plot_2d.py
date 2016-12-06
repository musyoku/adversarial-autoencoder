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

	x, _, label_ids = dataset.sample_labeled_data(images, labels, num_scatter, config.ndim_x, config.ndim_y)
	z = aae.to_numpy(aae.encode_x_z(x, test=True))
	visualizer.plot_labeled_z(z, label_ids, dir=args.plot_dir)
	
if __name__ == "__main__":
	main()
