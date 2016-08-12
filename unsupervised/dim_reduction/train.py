# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
from chainer import cuda, Variable
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
import util
from args import args
from model import conf, aae
import sampler

max_epoch = 1000
num_trains_per_epoch = 1000
batchsize = 100
n_steps_to_optimize_dis = 1

dataset = util.load_images(args.train_image_dir)

def sample_data():
	return util.sample_x_variable(batchsize, conf.ndim_x, dataset, gpu_enabled=conf.gpu_enabled)
	
total_time = 0
for epoch in xrange(1, max_epoch + 1):

	sum_loss_autoencoder = 0
	sum_loss_discriminator = 0
	sum_loss_generator = 0
	sum_loss_cluster_head = 0
	epoch_time = time.time()

	for t in xrange(1, num_trains_per_epoch + 1):
		# reconstruction phase
		x = sample_data()
		aae.update_learning_rate(conf.learning_rate_for_reconstruction_cost)
		sum_loss_autoencoder += aae.train_autoencoder_unsupervised(x)

		# regularization phase
		## train discriminator
		aae.update_learning_rate(conf.learning_rate_for_adversarial_cost)
		loss_discriminator = 0
		for k in xrange(n_steps_to_optimize_dis):
			if k > 0:
				x = sample_data()
			z_true = sampler.gaussian(batchsize, conf.ndim_z)
			y_true = sampler.onehot_categorical(batchsize, conf.ndim_y)
			loss_discriminator += aae.train_discriminator_yz(x, y_true, z_true)
		loss_discriminator /= n_steps_to_optimize_dis
		sum_loss_discriminator += loss_discriminator

		## train generator
		sum_loss_generator += aae.train_generator_x_yz(x)

		# train distance between every two cluster heads
		aae.update_learning_rate(conf.learning_rate_for_cluster_head)
		sum_loss_cluster_head += aae.train_cluster_head()

		if t % 10 == 0:
			sys.stdout.write("\rTraining in progress...({} / {})".format(t, num_trains_per_epoch))
			sys.stdout.flush()

	sys.stdout.write("\n")
	epoch_time = time.time() - epoch_time
	total_time += epoch_time
	print "epoch:", epoch
	print "  loss:"
	print "    autoencoder  : {:.4f}".format(sum_loss_autoencoder / num_trains_per_epoch)
	print "    discriminator: {:.4f}".format(loss_discriminator / num_trains_per_epoch)
	print "    generator    : {:.4f}".format(sum_loss_generator / num_trains_per_epoch)
	print "    cluster      : {:.4f}".format(sum_loss_cluster_head / num_trains_per_epoch)
	print "  time: {} min".format(int(epoch_time / 60)), "total: {} min".format(int(total_time / 60))
	aae.save(args.model_dir)
