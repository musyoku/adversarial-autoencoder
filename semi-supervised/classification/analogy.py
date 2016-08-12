# -*- coding: utf-8 -*-
import os, sys, time, pylab
import numpy as np
from chainer import cuda, Variable
import matplotlib.patches as mpatches
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
import util
from args import args
from model import conf, aae

try:
	os.mkdir(args.vis_dir)
except:
	pass

dataset, labels = util.load_labeled_images(args.test_image_dir)

n_analogies = 10
n_image_channels = 1
image_width = 28
image_height = 28
x, y, label_ids = util.sample_x_and_label_variables(n_analogies, conf.ndim_x, conf.ndim_y, dataset, labels, gpu_enabled=conf.gpu_enabled)
_, z = aae.generator_x_yz(x, test=True, apply_f=True)

if n_image_channels == 1:
	pylab.gray()
xp = np
if conf.gpu_enabled:
	x.to_cpu()
	xp = cuda.cupy
for m in xrange(n_analogies):
	pylab.subplot(n_analogies, conf.ndim_y + 2, m * 12 + 1)
	if n_image_channels == 1:
		pylab.imshow(x.data[m].reshape((image_width, image_height)), interpolation="none")
	elif n_image_channels == 3:
		pylab.imshow(x.data[m].reshape((n_image_channels, image_width, image_height)), interpolation="none")
	pylab.axis("off")
analogy_y = xp.identity(conf.ndim_y, dtype=xp.float32)
analogy_y = Variable(analogy_y)
for m in xrange(n_analogies):
	base_z = xp.empty((conf.ndim_y, z.data.shape[1]), dtype=xp.float32)
	for n in xrange(conf.ndim_y):
		base_z[n] = z.data[m]
	base_z = Variable(base_z)
	_x = aae.decoder_yz_x(analogy_y, base_z, test=True, apply_f=True)
	if conf.gpu_enabled:
		_x.to_cpu()
	for n in xrange(conf.ndim_y):
		pylab.subplot(n_analogies, conf.ndim_y + 2, m * 12 + 3 + n)
		if n_image_channels == 1:
			pylab.imshow(_x.data[n].reshape((image_width, image_height)), interpolation="none")
		elif n_image_channels == 3:
			pylab.imshow(_x.data[n].reshape((n_image_channels, image_width, image_height)), interpolation="none")
		pylab.axis("off")

fig = pylab.gcf()
fig.set_size_inches(n_analogies, conf.ndim_y)
pylab.savefig("{}/analogy.png".format(args.vis_dir))

