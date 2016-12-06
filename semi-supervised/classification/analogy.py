import numpy as np
import os, pylab
from model import aae
from args import args
import dataset

try:
	os.mkdir(args.plot_dir)
except:
	pass

def main():
	# load MNIST images
	images, labels = dataset.load_test_images()

	# config
	config = aae.config

	# settings
	num_analogies = 10
	pylab.gray()

	# generate style vector z
	x = dataset.sample_unlabeled_data(images, num_analogies, config.ndim_x, binarize=False)
	_, z = aae.encode_x_yz(x, apply_softmax=True)
	z = aae.to_numpy(z)

	# plot original image on the left
	for m in xrange(num_analogies):
		pylab.subplot(num_analogies, config.ndim_y + 2, m * 12 + 1)
		pylab.imshow(x[m].reshape((28, 28)), interpolation="none")
		pylab.axis("off")

	all_y = np.identity(config.ndim_y, dtype=np.float32)
	for m in xrange(num_analogies):
		# copy z as many as the number of classes
		fixed_z = np.repeat(z[m].reshape(1, -1), config.ndim_y, axis=0)
		gen_x = aae.to_numpy(aae.decode_yz_x(all_y, fixed_z))
		# plot images generated from each label
		for n in xrange(config.ndim_y):
			pylab.subplot(num_analogies, config.ndim_y + 2, m * 12 + 3 + n)
			pylab.imshow(gen_x[n].reshape((28, 28)), interpolation="none")
			pylab.axis("off")

	fig = pylab.gcf()
	fig.set_size_inches(num_analogies, config.ndim_y)
	pylab.savefig("{}/analogy.png".format(args.plot_dir))
	
if __name__ == "__main__":
	main()
