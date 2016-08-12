# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
from chainer import cuda, Variable
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
import util
from args import args
from model import conf, aae
import sampler

dataset, labels = util.load_labeled_images(args.train_image_dir)
max_epoch = 1000
num_trains_per_epoch = 1000
batchsize = 100
n_steps_to_optimize_dis = 1
n_labels = 10

def sample_data():
	return util.sample_x_and_label_variables(batchsize, conf.ndim_x, conf.ndim_y, dataset, labels, gpu_enabled=conf.gpu_enabled)

total_time = 0
for epoch in xrange(1, max_epoch + 1):

	sum_loss_autoencoder = 0
	sum_loss_discriminator = 0
	sum_loss_generator = 0
	epoch_time = time.time()

	for t in xrange(1, num_trains_per_epoch + 1):
		# train autoencoder
		x, y, label_ids = sample_data()
		sum_loss_autoencoder += aae.train_autoencoder(x, y)

		# train discriminator
		loss_discriminator = 0
		for k in xrange(n_steps_to_optimize_dis):
			if k > 0:
				x, y, label_ids = sample_data()
			z_true = sampler.gaussian(batchsize, conf.ndim_z)
			loss_discriminator += aae.train_discriminator_z(x, z_true)
		loss_discriminator /= n_steps_to_optimize_dis
		sum_loss_discriminator += loss_discriminator

		# train generator
		sum_loss_generator += aae.train_generator_x_z(x)

		if t % 10 == 0:
			sys.stdout.write("\rTraining in progress...({} / {})".format(t, num_trains_per_epoch))
			sys.stdout.flush()
	sys.stdout.write("\n")
	epoch_time = time.time() - epoch_time
	total_time += epoch_time
	print "epoch:", epoch
	print "  loss:"
	print "    autoencoder  : {:.4f}".format(sum_loss_autoencoder / num_trains_per_epoch)
	print "    discriminator: {:.4f}".format(sum_loss_discriminator / num_trains_per_epoch)
	print "    generator    : {:.4f}".format(sum_loss_generator / num_trains_per_epoch)
	print "  time: {} min".format(int(epoch_time / 60)), "total: {} min".format(int(total_time / 60))
	aae.save(args.model_dir)
