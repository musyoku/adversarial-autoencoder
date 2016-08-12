# -*- coding: utf-8 -*-
import os, sys, time, pylab
import numpy as np
from chainer import cuda, Variable
import matplotlib.patches as mpatches
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
import util
import sampler
import visualizer
from args import args
from model import conf, aae

try:
	os.mkdir(args.vis_dir)
except:
	pass

dataset, labels = util.load_labeled_images(args.test_image_dir)

num_images = len(dataset)
x, y, label_ids = util.sample_x_and_label_variables(num_images, conf.ndim_x, 10, dataset, labels, gpu_enabled=False)
if conf.gpu_enabled:
	y.to_gpu()
	x.to_gpu()

z = aae.generator_x_z(x, test=True, apply_f=True)
_x = aae.decoder_yz_x(y, z, test=True, apply_f=True)
if conf.gpu_enabled:
	z.to_cpu()
	_x.to_cpu()

visualizer.tile_x(_x.data, dir=args.vis_dir)
visualizer.plot_labeled_z(z.data, label_ids.data, dir=args.vis_dir)
