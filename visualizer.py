import pylab
import numpy as np
from PIL import Image
import matplotlib.patches as mpatches

def tile_images(image_batch, image_width=28, image_height=28, image_channel=1, dir=None, filename="images"):
	if dir is None:
		raise Exception()
	try:
		os.mkdir(dir)
	except:
		pass
	fig = pylab.gcf()
	fig.set_size_inches(16.0, 16.0)
	pylab.clf()
	pylab.gray()
	for m in range(100):
		pylab.subplot(10, 10, m + 1)
		pylab.imshow(image_batch[m].reshape((image_width, image_height)), interpolation="none")
		pylab.axis("off")
	pylab.savefig("{}/{}.png".format(dir, filename))

def plot_z(z_batch, dir=None, filename="z"):
	if dir is None:
		raise Exception()
	try:
		os.mkdir(dir)
	except:
		pass
	fig = pylab.gcf()
	fig.set_size_inches(20.0, 16.0)
	pylab.clf()
	for n in xrange(z_batch.shape[0]):
		result = pylab.scatter(z_batch[n, 0], z_batch[n, 1], s=40, marker="o", edgecolors='none')
	pylab.xlabel("z1")
	pylab.ylabel("z2")
	pylab.savefig("{}/{}.png".format(dir, filename))

def plot_labeled_z(z_batch, label_batch, dir=None, filename="labeled_z"):
	fig = pylab.gcf()
	fig.set_size_inches(20.0, 16.0)
	pylab.clf()
	colors = ["#2103c8", "#0e960e", "#e40402","#05aaa8","#ac02ab","#aba808","#151515","#94a169", "#bec9cd", "#6a6551"]
	for n in xrange(z_batch.shape[0]):
		result = pylab.scatter(z_batch[n, 0], z_batch[n, 1], c=colors[label_batch[n]], s=40, marker="o", edgecolors='none')

	classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
	recs = []
	for i in range(0, len(colors)):
		recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=colors[i]))

	ax = pylab.subplot(111)
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	ax.legend(recs, classes, loc="center left", bbox_to_anchor=(1.1, 0.5))
	pylab.xticks(pylab.arange(-4, 5))
	pylab.yticks(pylab.arange(-4, 5))
	pylab.xlabel("z1")
	pylab.ylabel("z2")
	pylab.savefig("{}/{}.png".format(dir, filename))
