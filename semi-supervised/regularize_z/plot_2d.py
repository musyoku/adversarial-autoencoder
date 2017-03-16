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
	x, _, label_ids = dataset.sample_labeled_data(images, labels, num_scatter)
	z = aae.to_numpy(aae.encode_x_z(x, test=True))
	plot.scatter_labeled_z(z, label_ids, dir=args.plot_dir)
	
if __name__ == "__main__":
	main()
